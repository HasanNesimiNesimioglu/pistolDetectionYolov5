from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render

# Create your views here.
from .yolov5 import detect


def index(request):
    return render(request, 'index.html', context={})


def start(request):
    detect.startStop = True
    return StreamingHttpResponse(detect.run(), content_type="multipart/x-mixed-replace;boundary=frame")


def stop(request):
    detect.startStop = False
    return HttpResponse(status=200)
