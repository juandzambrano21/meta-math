# reasoning/lean_engine.py

import os
import subprocess
import docker
import paramiko
import tempfile
from typing import Any, List, Tuple, Optional, Dict

class LeanResult:
    """
    Represents the result of executing a Lean code snippet.
    
    Attributes:
        value (Dict[str, str]): A dictionary containing the output of the Lean execution.
    """
    def __init__(self, value: Dict[str, str]) -> None:
        """
        Initializes a new LeanResult instance.

        Args:
            value (Dict[str, str]): The result output of the Lean code execution.
        """
        self.value = value

class LeanEngine:
    """
    Engine for executing Lean code within a Docker container, providing SSH access for execution.

    Attributes:
        ssh_host (str): The SSH host, defaulting to 'localhost'.
        ssh_port (int): The SSH port, defaulting to 2222.
        ssh_user (str): The SSH username, defaulting to 'root'.
        ssh_key_path (str): The path to the SSH private key, defaulting to '~/.ssh/id_rsa'.
        docker_client (docker.DockerClient): The Docker client used to manage containers.
        container (docker.models.containers.Container): The Docker container used for executing Lean code.
    """
    
    def __init__(
        self, 
        ssh_host: str = 'localhost', 
        ssh_port: int = 2222, 
        ssh_user: str = 'root', 
        ssh_key_path: str = '~/.ssh/id_rsa'
    ) -> None:
        """
        Initializes the LeanEngine with SSH and Docker configurations.

        Args:
            ssh_host (str): The SSH host, defaulting to 'localhost'.
            ssh_port (int): The SSH port, defaulting to 2222.
            ssh_user (str): The SSH username, defaulting to 'root'.
            ssh_key_path (str): The path to the SSH private key, defaulting to '~/.ssh/id_rsa'.
        """
        self.ssh_host: str = ssh_host
        self.ssh_port: int = ssh_port
        self.ssh_user: str = ssh_user
        self.ssh_key_path: str = os.path.expanduser(ssh_key_path)
        self.docker_client: docker.DockerClient = docker.from_env()
        self.container: docker.models.containers.Container = self._ensure_container()
        self._setup_ssh()

    def id(self) -> str:
        """
        Returns the identifier for the engine.

        Returns:
            str: The identifier of the LeanEngine, 'lean4'.
        """
        return 'lean4'

    def _ensure_container(self) -> docker.models.containers.Container:
        """
        Ensures the Docker container for Lean execution exists, creating it if necessary.

        Returns:
            docker.models.containers.Container: The Docker container instance used for Lean code execution.
        """
        container_name: str = "lean-container"
        
        try:
            existing_container: docker.models.containers.Container = self.docker_client.containers.get(container_name)
            print(f"Existing container '{container_name}' found. Removing it.")
            existing_container.remove(force=True)
        except docker.errors.NotFound:
            print(f"No existing container named '{container_name}' found. Proceeding to create a new one.")

        dockerfile: str = """
        FROM ubuntu:20.04

        ENV DEBIAN_FRONTEND=noninteractive
        RUN apt-get update && apt-get install -y curl git ssh sudo build-essential

        RUN useradd -ms /bin/bash lean
        RUN echo 'lean:lean' | chpasswd && adduser lean sudo

        USER lean
        WORKDIR /home/lean

        RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
        ENV PATH="/home/lean/.elan/bin:${PATH}"
        RUN elan default stable

        RUN mkdir /home/lean/.ssh

        EXPOSE 22
        CMD ["/usr/sbin/sshd", "-D"]
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w") as temp_dockerfile:
                temp_dockerfile.write(dockerfile)
            
            print("Building Docker image 'lean4-container-image'...")
            image, _ = self.docker_client.images.build(path=temp_dir, dockerfile="Dockerfile", tag="lean4-container-image")
        
            print("Running Docker container...")
            container: docker.models.containers.Container = self.docker_client.containers.run(
                image.id, 
                detach=True,
                name=container_name,
                ports={'22/tcp': self.ssh_port},
                user="lean",
                environment={"USER": "lean"},
                tty=True
            )
            return container

    def _setup_ssh(self) -> None:
        """
        Sets up SSH access to the Docker container, including generating an SSH key pair if necessary,
        and configuring the container to accept SSH connections using the generated key.
        """
        if not os.path.exists(self.ssh_key_path):
            print(f"SSH key not found at '{self.ssh_key_path}'. Generating a new SSH key pair.")
            subprocess.run(['ssh-keygen', '-t', 'rsa', '-b', '2048', '-f', self.ssh_key_path, '-N', ''], check=True)
        else:
            print(f"Using existing SSH key at '{self.ssh_key_path}'.")

        print("Configuring SSH inside the Docker container...")
        # Ensure .ssh directory exists
        subprocess.run(['docker', 'exec', self.container.id, 'mkdir', '-p', '/home/lean/.ssh'], check=True)
        # Copy the public key to authorized_keys
        subprocess.run(['docker', 'cp', f'{self.ssh_key_path}.pub', f'{self.container.id}:/home/lean/.ssh/authorized_keys'], check=True)
        # Set appropriate permissions
        subprocess.run(['docker', 'exec', self.container.id, 'chmod', '600', '/home/lean/.ssh/authorized_keys'], check=True)
        subprocess.run(['docker', 'exec', self.container.id, 'chown', 'lean:lean', '/home/lean/.ssh/authorized_keys'], check=True)
        # Start SSH service
        subprocess.run(['docker', 'exec', '-u', 'root', self.container.id, 'service', 'ssh', 'start'], check=True)
        print("SSH setup completed.")

    def forward(self, code: str) -> Tuple[List[LeanResult], dict]:
        """
        Executes Lean code provided as a string.

        Args:
            code (str): The Lean code to be executed.

        Returns:
            Tuple[List[LeanResult], dict]: A tuple containing the result of the Lean execution and associated metadata.
        """
        rsp: Optional[LeanResult] = None
        err: Optional[str] = None
        tmpfile_path: Optional[str] = None
        metadata: Dict[str, Any] = {}
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".lean") as tmpfile:
                tmpfile.write(code.encode())
                tmpfile_path = tmpfile.name

            print(f"Executing Lean code from temporary file '{tmpfile_path}'...")
            output, exec_metadata = self._execute_lean(tmpfile_path)
            metadata.update(exec_metadata)

            if output:
                rsp = LeanResult({'output': output})
            else:
                metadata['status'] = 'no_output'
        except Exception as e:
            err = str(e)
            metadata.update({'status': 'error', 'message': err})
            print(f"Error during Lean execution: {err}")
        finally:
            if tmpfile_path and os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)
                print(f"Removed temporary file '{tmpfile_path}'.")
            # Do not remove the container here; keep it running for further executions

        return [rsp] if rsp else [], metadata

    def _execute_lean(self, filepath: str) -> Tuple[str, dict]:
        """
        Executes a Lean script within the Docker container via SSH.

        Args:
            filepath (str): The path to the Lean file to be executed.

        Returns:
            Tuple[str, dict]: The output from the Lean execution and associated status metadata.
        """
        try:
            print("Establishing SSH connection to the Docker container...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ssh_host, port=self.ssh_port, username=self.ssh_user, key_filename=self.ssh_key_path)
            print("SSH connection established.")

            print(f"Uploading Lean file '{filepath}' to the container...")
            sftp = ssh.open_sftp()
            remote_path: str = f"/home/lean/{os.path.basename(filepath)}"
            sftp.put(filepath, remote_path)
            sftp.close()
            print(f"Lean file uploaded to '{remote_path}'.")

            print(f"Executing Lean file '{remote_path}'...")
            stdin, stdout, stderr = ssh.exec_command(f"lean {remote_path}")
            output = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                print("Lean execution error:", error)

            print("Removing remote Lean file...")
            ssh.exec_command(f"rm {remote_path}")
            ssh.close()

            if "error" in output.lower() or "error" in error.lower():
                return output + error, {'status': 'failure'}
            elif not output and not error:
                return "Lean program halted successfully with no output.", {'status': 'success'}
            else:
                return output, {'status': 'success'}

        except Exception as e:
            raise RuntimeError(f"SSH command execution failed: {str(e)}")
