from django.core.signals import request_started, request_finished
from django.dispatch import receiver



model_session = None

@receiver(request_started)
def request_started_handler(sender, **kwargs):
    # Perform actions when a request starts (if needed)
    pass

@receiver(request_finished)
def request_finished_handler(sender, **kwargs):
    # Close the TensorFlow session when the request is finished
    global model_session
    if model_session:
        model_session.close()
        print("TF session closed")
