from django.apps import AppConfig



class HerbalookAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'herbaLook_app'

    # Import signals module when the app is ready
    def ready(self):
        import herbaLook_app.signals  
