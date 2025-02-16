let currentVideo = null;

const endpoint = 'https://0.0.0.0:8000';

// get video element
const video = document.getElementById('video');

function upload() {
    // create file popup
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/mp4';
    input.onchange = function() {
        const file = input.files[0];
        const url = URL.createObjectURL(file);
        video.src = url;
        currentVideo = file;
    }
    input.click();
}

function analyze() {
    if (currentVideo === null) {
        alert('Please upload a video first');
        return;
    }
    const form = new FormData();
    form.append('video', currentVideo);
    // set video to loading
    document.getElementById('loader').style.display = 'flex';
    fetch(endpoint + '/process', {
        method: 'POST',
        body: form
    }).then(response => {
        document.getElementById('loader').style.display = 'none';
        if (response.ok) {
            video.src = endpoint + '/result/0/video'
        } else {
            throw new Error('Failed to process video');
        }
    });
}