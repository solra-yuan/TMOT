import visdom

class VisdomSingleton:
    _instance = None

    @staticmethod
    def get_instance(settings=None):
        if VisdomSingleton._instance is None:
            # Create a new instance if one does not exist yet
            if settings is None:
                raise ValueError("Settings must be provided to initialize the Visdom instance.")
            VisdomSingleton._instance = visdom.Visdom(
                server=settings.get('server', 'http://localhost'),
                port=settings.get('port', 8097),
                env=settings.get('env', 'main')
            )
        return VisdomSingleton._instance

