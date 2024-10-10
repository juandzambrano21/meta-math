import os
import subprocess
import docker
import paramiko
import tempfile
import time
from typing import Any, List, Tuple, Optional, Dict

class CoqResult:
    """
    Represents the result of executing a Coq code snippet.
    
    Attributes:
        value (Dict[str, str]): A dictionary containing the output of the Coq execution.
    """
    def __init__(self, value: Dict[str, str]) -> None:
        """
        Initializes a new CoqResult instance.

        Args:
            value (Dict[str, str]): The result output of the Coq code execution.
        """
        self.value = value

class CoqEngine:
    """
    Engine for executing Coq code within a Docker container, providing SSH access for execution.

    Attributes:
        ssh_host (str): The SSH host, defaulting to 'localhost'.
        ssh_port (int): The SSH port, defaulting to 2222.
        ssh_user (str): The SSH username, defaulting to 'root'.
        ssh_key_path (str): The path to the SSH private key, defaulting to '~/.ssh/id_rsa'.
        docker_client (docker.DockerClient): The Docker client used to manage containers.
        container (docker.models.containers.Container): The Docker container used for executing Coq code.
    """
    
    def __init__(
        self, 
        ssh_host: str = 'localhost', 
        ssh_port: int = 2222, 
        ssh_user: str = 'root', 
        ssh_key_path: str = '~/.ssh/id_rsa'
    ) -> None:
        """
        Initializes the CoqEngine with SSH and Docker configurations.

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
            str: The identifier of the CoqEngine, 'coq'.
        """
        return 'coq'

    def _ensure_container(self) -> docker.models.containers.Container:
        """
        Ensures the Docker container for Coq execution exists, creating it if necessary.

        Returns:
            docker.models.containers.Container: The Docker container instance used for Coq code execution.
        """
        container_name: str = "coq-container"
        
        try:
            existing_container: docker.models.containers.Container = self.docker_client.containers.get(container_name)
            print(f"Existing container '{container_name}' found. Removing it.")
            existing_container.remove(force=True)
        except docker.errors.NotFound:
            print(f"No existing container named '{container_name}' found. Proceeding to create a new one.")

        # Use the provided Dockerfile content
        dockerfile: str = """
        FROM debian:bullseye-slim

        # Install necessary packages
        RUN apt-get update && apt-get install -y \
            wget \
            gnupg \
            openssh-server \
            sudo \
            opam \
            m4 \
            git \
            curl \
            rsync \
            ca-certificates \
            build-essential \
            unzip \
            pkg-config \
            libgmp-dev

        # Install OCaml and Coq
        RUN opam init -a --disable-sandboxing \
            && opam switch create 4.13.1 \
            && eval $(opam env) \
            && opam install -y coq.8.15.2

        # Set OPAM environment variables
        ENV OPAMYES=true
        ENV OPAMJOBS=2
        ENV OPAMVERBOSE=1
        ENV OPAMCOLOR=never
        ENV CAML_LD_LIBRARY_PATH=/root/.opam/4.13.1/lib/stublibs
        ENV OCAML_TOPLEVEL_PATH=/root/.opam/4.13.1/lib/toplevel
        ENV PATH=/root/.opam/4.13.1/bin:$PATH

        # Configure SSH
        RUN mkdir /var/run/sshd

        RUN echo 'root:root' | chpasswd

        RUN sed -i 's/#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config

        EXPOSE 22

        CMD ["/usr/sbin/sshd", "-D"]

       
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w") as temp_dockerfile:
                temp_dockerfile.write(dockerfile)
            
            print("Building Docker image 'coq-container-image'...")
            try:
                image, _ = self.docker_client.images.build(path=temp_dir, dockerfile="Dockerfile", tag="coq-container-image")
            except docker.errors.BuildError as e:
                print(f"Docker build failed: {e}")
                raise

            print("Running Docker container...")
            try:
                container: docker.models.containers.Container = self.docker_client.containers.run(
                    image.id, 
                    detach=True,
                    name=container_name,
                    ports={'22/tcp': self.ssh_port}
                )
            except Exception as e:
                print(f"Failed to run Docker container: {e}")
                raise

            # Wait for SSH to be ready
            time.sleep(5)

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
        subprocess.run(['docker', 'exec', self.container.id, 'mkdir', '-p', '/root/.ssh'], check=True)
        # Copy the public key to authorized_keys
        subprocess.run(['docker', 'cp', f'{self.ssh_key_path}.pub', f'{self.container.id}:/root/.ssh/authorized_keys'], check=True)
        # Set appropriate permissions
        subprocess.run(['docker', 'exec', self.container.id, 'chmod', '600', '/root/.ssh/authorized_keys'], check=True)
        subprocess.run(['docker', 'exec', self.container.id, 'chown', 'root:root', '/root/.ssh/authorized_keys'], check=True)
        print("SSH setup completed.")

    def forward(self, code: str) -> Tuple[List[CoqResult], dict]:
        """
        Executes Coq code provided as a string.

        Args:
            code (str): The Coq code to be executed.

        Returns:
            Tuple[List[CoqResult], dict]: A tuple containing the result of the Coq execution and associated metadata.
        """
        rsp: Optional[CoqResult] = None
        err: Optional[str] = None
        tmpfile_path: Optional[str] = None
        metadata: Dict[str, Any] = {}
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".v") as tmpfile:
                tmpfile.write(code.encode())
                tmpfile_path = tmpfile.name

            print(f"Executing Coq code from temporary file '{tmpfile_path}'...")
            output, exec_metadata = self._execute_coq(tmpfile_path)
            metadata.update(exec_metadata)

            if output:
                rsp = CoqResult({'output': output})
            else:
                metadata['status'] = 'no_output'
        except Exception as e:
            err = str(e)
            metadata.update({'status': 'error', 'message': err})
            print(f"Error during Coq execution: {err}")
        finally:
            if tmpfile_path and os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)
                print(f"Removed temporary file '{tmpfile_path}'.")
            # Do not remove the container here; keep it running for further executions

        return [rsp] if rsp else [], metadata

    def _execute_coq(self, filepath: str) -> Tuple[str, dict]:
        """
        Executes a Coq script within the Docker container via SSH.

        Args:
            filepath (str): The path to the Coq file to be executed.

        Returns:
            Tuple[str, dict]: The output from the Coq execution and associated status metadata.
        """
        try:
            print("Establishing SSH connection to the Docker container...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Retry logic for SSH connection
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    ssh.connect(self.ssh_host, port=self.ssh_port, username=self.ssh_user, key_filename=self.ssh_key_path)
                    print("SSH connection established.")
                    break
                except Exception as e:
                    print(f"SSH connection attempt {attempt + 1} failed: {e}")
                    time.sleep(2)
            else:
                raise RuntimeError("Failed to establish SSH connection after multiple attempts.")

            print(f"Uploading Coq file '{filepath}' to the container...")
            sftp = ssh.open_sftp()
            remote_path: str = f"/root/{os.path.basename(filepath)}"
            sftp.put(filepath, remote_path)
            sftp.close()
            print(f"Coq file uploaded to '{remote_path}'.")

            # Set OPAM environment variables in the SSH session
            opam_env_cmd = 'eval $(opam env) && '

            print(f"Executing Coq file '{remote_path}'...")
            stdin, stdout, stderr = ssh.exec_command(opam_env_cmd + f"coqc {remote_path}")
            output = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                print("Coq execution error:", error)

            print("Removing remote Coq file...")
            ssh.exec_command(f"rm {remote_path}")
            ssh.close()

            if "Error" in output or "Error" in error:
                return output + error, {'status': 'failure'}
            elif not output and not error:
                return "Coq program executed successfully with no output.", {'status': 'success'}
            else:
                return output + error, {'status': 'success'}

        except Exception as e:
            raise RuntimeError(f"SSH command execution failed: {str(e)}")
