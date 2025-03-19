from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class VisdomOptions:
    """
    A class to hold configuration options for Visdom. This class is immutable 
    (frozen=True), meaning once an instance is created, its attributes cannot be modified.

    Attributes:
        env (str): The environment name in Visdom (e.g., "main").
        port (int): The port number on which the Visdom server is running.
        server (str): The URL of the Visdom server (e.g., "http://localhost").
    """

    env: str  # The environment name in Visdom.
    port: int  # The port on which the Visdom server listens.
    server: str  # The Visdom server URL.

    def asdict(self):
        """
        Converts the current instance into a dictionary.

        Returns:
            dict: A dictionary representation of the current VisdomOptions instance.
        """
        return asdict(self)


class VisdomOptionSingleton:
    """
    Singleton class to manage a single instance of VisdomOptions. The singleton ensures that
    there is only one instance of the VisdomOptions throughout the application.

    Attributes:
        _instance (VisdomOptions): The singleton instance of VisdomOptions.
    """

    # Class-level variable to hold the singleton instance.
    _instance: VisdomOptions = None

    def __new__(cls, env: str = "main", port: int = 8097, server: str = 'http://localhost'):
        """
        Creates and returns the singleton instance of VisdomOptions.

        If the singleton instance does not exist, it is created using the provided `env`, `port`, and `server`.
        If no option is provided during the first instantiation, a ValueError is raised.

        Args:
            env (str): The environment name in Visdom.
            port (int): The port number on which the Visdom server is running.
            server (str): The URL of the Visdom server.          

        Returns:
            VisdomOptions: The singleton instance of the VisdomOptions class.
        """
        # If no instance exists, create one using the provided `option`
        if cls._instance is None:
            cls._instance = VisdomOptions(env=env, port=port, server=server)
        # Return the existing instance
        return cls._instance

    def get_visdom_options(self):
        """
        Retrieves the current singleton instance of VisdomOptions.

        Returns:
            VisdomOptions: The current instance of VisdomOptions.
        """
        return self._instance

    def set_visdom_options(self, env: str, port: int, server: str):
        """
        Sets a new VisdomOptions instance as the singleton. This allows updating the configuration 
        with new values for `env`, `port`, and `server`. Since VisdomOptions is a frozen (immutable) 
        dataclass, a new instance will be created rather than modifying the existing one.

        Args:
            env (str): The new environment name.
            port (int): The new port number.
            server (str): The new server URL.
        """
        # Update the singleton instance with new values
        self._instance = VisdomOptions(env=env, port=port, server=server)


if __name__ == '__main__':
    import visdom

    # Create an instance of VisdomOptions with configuration details.
    visdom_options = VisdomOptionSingleton(
        env='flir_demo_main', port=8097, server='http://localhost'
    )

    # Initialize a Visdom instance using the options from VisdomOptions.
    # Convert VisdomOptions to a dictionary and pass to Visdom.
    vis = visdom.Visdom(**visdom_options.asdict())
