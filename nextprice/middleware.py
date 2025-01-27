from django.shortcuts import redirect

class RedirectAuthenticatedUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated and request.path in ['/login/', '/signup/']:
            return redirect('prediction')  # Use the URL name from urls.py
        return self.get_response(request)
