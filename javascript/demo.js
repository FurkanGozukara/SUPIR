let isActive = false;
let openLabels = [];
let fullscreenButton;
let sliderLoaded = false;
let vidLength = 0;
let vidFps = 0;


function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function (id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

function getRealElement(selector, get_input = false) {
    let elem = gradioApp().getElementById(selector);
    let output = elem;
    if (elem) {
        let child = elem.querySelector('#' + selector);
        if (child) {
            output = child;
        }
    }
    if (get_input) {
        return output.querySelector('input');
    }
    return output;
}

function filterArgs(argsCount, arguments) {
    let args_out = [];
    if (arguments.length >= argsCount && argsCount !== 0) {
        for (let i = 0; i < argsCount; i++) {
            args_out.push(arguments[i]);
        }
    }
    return args_out;
}

function update_slider() {
    console.log("Input args: ", arguments);
    configureSlider(arguments[4], arguments[3]);
    return filterArgs(7, arguments);
}

function configureSlider(videoLength, fps) {
    if (vidFps === fps && vidLength === videoLength) {
        console.log('configureSlider', 'videoLength and fps are the same');
        return;
    }
    if (videoLength === 0) {
        console.log('configureSlider', 'videoLength is 0');
        return;
    }
    console.log('configureSlider, re-creating slider.', 'videoLength', videoLength, 'fps', fps);
    vidFps = fps;
    vidLength = videoLength;
    let connectSlider2 = document.getElementById('frameSlider');
    let endTimeLabel = document.getElementById('endTimeLabel');
    let nowTimeLabel = document.getElementById('nowTimeLabel');
    if (sliderLoaded) {
        try {
            connectSlider2.noUiSlider.destroy();
        } catch (e) {
            console.log('configureSlider', 'noUiSlider.destroy failed', e);
        }
    }


    let midPoint = Math.floor(videoLength / 2);

    noUiSlider.create(connectSlider2, {
        start: [0, midPoint, videoLength],
        connect: [false, true, true, false],
        range: {
            'min': 0,
            'max': videoLength
        }
    });
    videoLength = Math.floor(videoLength / fps);
    midPoint = Math.floor(midPoint / fps);

    endTimeLabel.innerHTML = formatSeconds(videoLength);
    nowTimeLabel.innerHTML = formatSeconds(midPoint);
    connectSlider2.noUiSlider.on('set', updateSliderElements);
    connectSlider2.noUiSlider.on('slide', updateSliderTimes);
    sliderLoaded = true;
}

function formatSeconds(seconds) {
    let minutes = Math.floor(seconds / 60);
    let remainingSeconds = seconds % 60;
    (remainingSeconds < 10) ? remainingSeconds = '0' + remainingSeconds : remainingSeconds = remainingSeconds.toString();
    if (minutes < 60) {
        let hours = Math.floor(minutes / 60);
        let remainingMinutes = minutes % 60;
        (remainingMinutes < 10) ? remainingMinutes = '0' + remainingMinutes : remainingMinutes = remainingMinutes.toString();
        return hours + ':' + remainingMinutes + ':' + remainingSeconds;
    }
    return minutes + ':' + remainingSeconds;
}

function updateSliderElements(values, handle, unencoded, tap, positions, noUiSlider) {
    console.log('updateSliderElements', values, handle);
    // Convert strings from values to floats
    let startTime = Math.floor(values[0]);
    let nowTime = Math.floor(values[1]);
    let endTime = Math.floor(values[2]);
    let start = document.getElementById('startTimeLabel');
    let end = document.getElementById('endTimeLabel');
    let now = document.getElementById('nowTimeLabel');

    let startNumber = getRealElement('start_time', true);
    let endNumber = getRealElement('end_time', true);
    let nowNumber = getRealElement('current_time', true);
    let fpsNumber = getRealElement('video_fps', true);
    let lastStartTime = startNumber.value;
    let lastNowTime = nowNumber.value;
    let lastEndTime = endNumber.value;
    startNumber.value = startTime;
    nowNumber.value = nowTime;
    endNumber.value = endTime;
    let times = [lastStartTime, lastNowTime, lastEndTime];
    let idx = 0;
    [startNumber, nowNumber, endNumber].forEach(el => {
        if (el.value !== times[idx]) {
            el.dispatchEvent(new Event('input', {'bubbles': true}));
        }
        idx++;
    });

    let fps = fpsNumber.value;
    startTime = Math.floor(startTime / fps);
    endTime = Math.floor(endTime / fps);
    nowTime = Math.floor(nowTime / fps);
    start.innerHTML = formatSeconds(startTime);
    end.innerHTML = formatSeconds(endTime);
    now.innerHTML = formatSeconds(nowTime);
}

function updateSliderTimes(values, handle, unencoded, tap, positions, noUiSlider) {
    console.log('updateSliderTimes', values, handle);
    // Convert strings from values to floats
    let startTime = Math.floor(values[0]);
    let nowTime = Math.floor(values[1]);
    let endTime = Math.floor(values[2]);
    let start = document.getElementById('startTimeLabel');
    let end = document.getElementById('endTimeLabel');
    let now = document.getElementById('nowTimeLabel');

    let fpsNumber = getRealElement('video_fps', true);
    let fps = fpsNumber.value;
    startTime = Math.floor(startTime / fps);
    endTime = Math.floor(endTime / fps);
    nowTime = Math.floor(nowTime / fps);
    start.innerHTML = formatSeconds(startTime);
    end.innerHTML = formatSeconds(endTime);
    now.innerHTML = formatSeconds(nowTime);
}

function toggleFullscreen() {
    console.log('toggleFullscreen', isActive);
    if (isActive) {
        // If active is true, enumerate openLables and add the .open class to each element
        openLabels.forEach((label) => {
            label.classList.add('open');
            // Get the sibling div and remove display: none!important
            let sibling = label.nextElementSibling;
            sibling.style.display = 'block';
        });
        openLabels = [];
        isActive = false;
    } else {
        openLabels = document.querySelectorAll('.label-wrap.open');
        openLabels.forEach((label) => {
            label.classList.remove('open');
            // Get the sibling div and add display: none!important
            let sibling = label.nextElementSibling;
            sibling.style.display = 'none';

        });
        isActive = true;
    }
    let output = filterArgs(0, arguments);
    console.log('toggleFullscreen', isActive, output);
    return output;
}

function downloadImage() {
    //.thumbnail-item.selected
    let selectedThumbnail = document.querySelector('.thumbnail-item.selected');
    let args;
    if (selectedThumbnail) {
        let img = selectedThumbnail.querySelector('img');
        if (img) {
            let url = img.src;
            let filename = url.split('/').pop();
            let path = url.split('/').slice(0, -1).join('/');
            args = {
                url: url,
                filename: filename,
                path: path
            };
            console.log('downloadImage', args);
        }
    } else {
        let gallery1 = document.getElementById('gallery1');
        // Get the second img, which is a child somewhere in gallery1
        let img = gallery1.querySelectorAll('img');
        if (img.length > 1) {
            let url = img[1].src;
            let filename = url.split('/').pop();
            let path = url.split('/').slice(0, -1).join('/');
            args = {
                url: url,
                filename: filename,
                path: path
            };
            console.log('downloadImage', args);
        }

    }
    if (args.hasOwnProperty('url')) {
        console.log('downloadImage (url)', args.url, args.filename);
        let url = args.url;
        if (url.length > 0) {
            fetch(url)
                .then(response => response.blob())
                .then(blob => {
                    let link = document.createElement('a');
                    link.href = window.URL.createObjectURL(blob);
                    if (args.hasOwnProperty('path')) {
                        let path = args.path;
                        let filename = path.split('/').pop();
                        if (filename.length > 0) {
                            link.download = filename;
                        }
                    }
                    // This is necessary as link.click() does not work on the latest firefox
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    window.URL.revokeObjectURL(link.href); // Clean up the URL object
                })
                .catch(console.error);
        }
    }

}

document.addEventListener('keydown', (event) => {
    if (isActive && event.key === 'Escape') {
        console.log('Escape key pressed');
        fullscreenButton = getRealElement('fullscreen_button');
        // Click the fullscreen button to close the fullscreen mode
        fullscreenButton.click();
    }
    if (event.key === 'Escape') {
        let openInfoDivs = document.querySelectorAll('.info-btn.open');
        openInfoDivs.forEach((openDiv) => {
            openDiv.classList.remove('open');
            let infoDiv = openDiv.nextElementSibling;
            infoDiv.style.display = 'none';
        });
    }
});

document.addEventListener('click', (event) => {
    // If the target has the class "image-button"
    if (event.target.classList.contains('image-button')) {
        // Get the child img object
        let img = event.target.querySelector('img');
        // Toggle the "fullscreen" class on the img object
        img.classList.toggle('full_screen');
    }
    if (!event.target.classList.contains('info-btn')) {
        let openInfoDivs = document.querySelectorAll('.info-btn.open');
        openInfoDivs.forEach((openDiv) => {
            openDiv.classList.remove('open');
            let infoDiv = openDiv.nextElementSibling;
            infoDiv.style.display = 'none';
        });
    }
});

// Document ready
document.addEventListener('DOMContentLoaded', () => {
    // Get the fullscreen button
    console.log('DOMContentLoaded');
    addInfoButtons();
});

setTimeout(() => {
    addInfoButtons();
}, 2000);


function addInfoButtons() {
    let infoButtons = document.querySelectorAll('.info-button');
    console.log('addInfoButtons', infoButtons.length, infoButtons);
    infoButtons.forEach((button) => {
        let id = button.id;
        console.log('addInfoButtons', id);
        // Check if the id is in elementsConfig
        if (elementsConfig.hasOwnProperty(id)) {
            let config = elementsConfig[id];
            let showInfoButton = config.showInfoButton;
            let hasRefresh = config.hasOwnProperty('hasRefresh') ? config.hasRefresh : false;
            // See if there is an input of the type range in the button
            let rangeInput = button.querySelector('input[type="range"]');
            if (rangeInput) {
                showInfoButton = false;
            }
            if (showInfoButton) {
                let div = document.createElement('div');
                div.classList.add('info-btn');
                div.innerHTML = '<i></i>';
                if (hasRefresh) {
                    div.style.right = '40px';
                    div.style.top = '10px';
                }
                button.appendChild(div);
                let infoDiv = document.createElement('div');
                infoDiv.classList.add('info-title-div');
                infoDiv.style.display = 'none';
                infoDiv.innerHTML = config.title;
                button.appendChild(infoDiv);
                div.addEventListener('click', (event) => {
                    if (div.classList.contains('open')) {
                        div.classList.remove('open');
                        let infoDiv = div.nextElementSibling;
                        infoDiv.style.display = 'none';
                    } else {
                        // Close all other open info-divs
                        let openInfoDivs = document.querySelectorAll('.info-btn.open');
                        openInfoDivs.forEach((openDiv) => {
                            openDiv.classList.remove('open');
                            let infoDiv = openDiv.nextElementSibling;
                            infoDiv.style.display = 'none';
                        });
                        div.classList.add('open');
                        let infoDiv = div.nextElementSibling;
                        let mouseX = event.clientX;
                        let mouseY = event.clientY;
                        let infoDivWidth = 500;
                        let infoDivHeight = 100;
                        let windowWidth = window.innerWidth;
                        let windowHeight = window.innerHeight;
                        let left = mouseX + infoDivWidth > windowWidth ? mouseX - infoDivWidth : mouseX;
                        let top = mouseY + infoDivHeight > windowHeight ? mouseY - infoDivHeight : mouseY;
                        infoDiv.style.left = left + 'px';
                        infoDiv.style.top = top + 'px';
                        infoDiv.style.position = 'fixed';
                        infoDiv.style.display = 'block';
                    }
                });
            } else {
                let title = config.title;
                button.setAttribute('title', title);

            }
        }
    });
}

const elementsConfig = {
    "a_prompt": {
        title: "This is the default positive prompt. It will be applied to SUPIR captioning and combined with the caption or main prompt if provided.",
        showInfoButton: true
    },
    "ae_dtype": {
        title: "Specifies the data type for auto-encoder processing. Choose between different precision options.",
        showInfoButton: true
    },
    "apply_bg": {
        title: "This will apply background correction to the image during SUPIR upscaling.",
        showInfoButton: true
    },
    "apply_face": {title: "This will apply face restoration during SUPIR upscaling.", showInfoButton: true},
    "apply_llava": {
        title: "This will apply LLaVa captioning to the input. NOT recommended for video.",
        showInfoButton: true
    },
    "apply_supir": {title: "Process the image using SUPIR with the selected settings.", showInfoButton: true},
    "auto_unload_llava": {
        title: "Enable this to automatically unload LLaVa after processing to free up resources.",
        showInfoButton: true
    },
    "batch_process_folder": {
        title: "Specify the folder for batch processing. All eligible files in this folder will be processed.",
        showInfoButton: true
    },
    "ckpt_select": {
        title: "Select the checkpoint to use for model inference. Different checkpoints might produce varied stylizations.",
        showInfoButton: true,
        hasRefresh: true
    },
    "checkpoint_type": {
        title: "Select the type of checkpoint to use for processing (SDXL/Lightning). This will auto-configure the necessary parameters for the model type.",
        showInfoButton: true
    },
    "color_fix_type": {
        title: "Choose the type of color correction to apply during processing. Options might include 'None', 'AdaLn', 'Wavelet'.",
        showInfoButton: true
    },
    "diff_dtype": {
        title: "Select the data type for differential processing, affecting performance and memory usage.",
        showInfoButton: true
    },
    "edm_steps": {
        title: "Set the number of EDM steps for the process. Higher values might improve quality at the cost of processing time.",
        showInfoButton: true
    },
    "face_prompt": {
        title: "Enter a prompt to guide the face restoration process, influencing the final outcome's characteristics.",
        showInfoButton: true
    },
    "face_resolution": {
        title: "Adjust the resolution for face processing. Higher values improve detail but increase processing time.",
        showInfoButton: true
    },
    "linear_CFG": {
        title: "Toggle the linear CFG adjustment. This might affect the stylistic elements of the generated images.",
        showInfoButton: true
    },
    "linear_s_stage2": {
        title: "Adjust the second stage of linear S processing. Fine-tune to balance between style and fidelity.",
        showInfoButton: true
    },
    "main_prompt": {
        title: "Enter the main prompt for image generation. This will guide the overall theme and content of the output.",
        showInfoButton: true
    },
    "make_comparison_video": {
        title: "Enable this to generate a comparison video between original and processed outputs.",
        showInfoButton: true
    },
    "model_select": {
        title: "Select the model to use for processing. V0Q emphasizes Quality, V0F emphasized Fidelity.",
        showInfoButton: true
    },
    "n_prompt": {
        title: "Enter a negative prompt to guide what the model should avoid in the output.",
        showInfoButton: true
    },
    "num_images": {
        title: "Set the number of images to generate. Increasing this number will proportionally increase processing time.",
        showInfoButton: true
    },
    "num_samples": {
        title: "Specify the number of samples to use in the process. Higher numbers might improve quality at the expense of speed.",
        showInfoButton: true
    },
    "output_video_format": {
        title: "Select the format for the output video. Options are MP4 and MKV.",
        showInfoButton: true
    },
    "output_video_quality": {
        title: "Adjust the quality of the output video. Higher values improve clarity but increase file size.",
        showInfoButton: true
    },
    "outputs_folder": {
        title: "Specify the folder where output files will be saved. Ensure this location has enough space to accommodate the files.",
        showInfoButton: true
    },
    "qs": {title: "This is the prompt used for LLaVa captioning.", showInfoButton: true},
    "random_seed": {
        title: "Toggle to use a random seed for each process, ensuring each output is unique.",
        showInfoButton: true
    },
    "s_cfg": {
        title: "Configure the S parameter for CFG adjustment, influencing the creative freedom of the model.",
        showInfoButton: true
    },
    "s_churn": {
        title: "Adjust the churn setting, affecting how much the model deviates from the initial input.",
        showInfoButton: true
    },
    "s_noise": {
        title: "Set the noise level for the process. Noise can add variability but might reduce clarity.",
        showInfoButton: true
    },
    "s_stage1": {
        title: "Configure the first stage S parameter, balancing between initial guidance and model interpretation.",
        showInfoButton: true
    },
    "s_stage2": {
        title: "Configure the second stage S parameter, fine-tuning the balance between guidance and interpretation.",
        showInfoButton: true
    },
    "sampler": {
        title: "Select the sampling method for the model. Different samplers can affect the style and coherence of the output.",
        showInfoButton: true
    },
    "save_captions": {
        title: "Enable this to save captions alongside images, useful for keeping track of prompts and settings.",
        showInfoButton: true
    },
    "seed": {
        title: "Set a specific seed for reproducibility. This ensures the same input will produce the same output.",
        showInfoButton: true
    },
    "spt_linear_CFG": {
        title: "Adjust the CFG for SPT linear processing. This might affect certain stylistic aspects of the output.",
        showInfoButton: true
    },
    "spt_linear_s_stage2": {
        title: "Fine-tune the second stage of SPT linear processing for a balance between style and fidelity.",
        showInfoButton: true
    },
    "src_file": {
        title: "Select the source file for processing. This file will be the basis for all subsequent operations.",
        showInfoButton: true
    },
    "temperature": {
        title: "Adjust the temperature for the process. Higher temperatures might increase creativity at the risk of coherence.",
        showInfoButton: true
    },
    "top_p": {
        title: "Set the top P value, controlling the diversity of the generated content by limiting the sampling pool.",
        showInfoButton: true
    },
    "upscale": {
        title: "Adjust the upscale factor for the output. Higher values increase resolution but might introduce artifacts.",
        showInfoButton: false
    },
    "video_duration": {
        title: "Specify the duration for the output video. This controls how long the final video will be.",
        showInfoButton: true
    },
    "video_end": {
        title: "Set the end time for video processing. This determines where the video will stop.",
        showInfoButton: true
    },
    "video_fps": {
        title: "Set the frames per second for the output video. Higher FPS results in smoother video at the cost of larger file sizes.",
        showInfoButton: true
    },
    "video_height": {
        title: "Set the height for the output video. This determines the vertical resolution.",
        showInfoButton: true
    },
    "video_start": {
        title: "Set the start time for video processing. This determines where the video will begin.",
        showInfoButton: true
    },
    "video_width": {
        title: "Set the width for the output video. This determines the horizontal resolution.",
        showInfoButton: true
    },
    "prompt_style": {
        title: "Choose the style for prompts. This might affect how the prompts are interpreted by the model.",
        showInfoButton: true,
        hasRefresh: true
    },
    "btn_open_outputs": {
        title: "Button to open the outputs folder. Use this to quickly access your generated content.",
        showInfoButton: true
    }
};

