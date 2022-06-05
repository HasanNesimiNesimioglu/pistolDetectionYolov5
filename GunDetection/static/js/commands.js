function startDetect() {
    document.getElementById("stream").src = "start";
}

function stopDetect() {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", 'stop', false); // false for synchronous request
    xmlHttp.send(null);
    document.getElementById("stream").src = null;

}