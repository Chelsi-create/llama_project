// Global state management
const cameraStreams = {};
let isLiveClassification = false;
let classificationInterval = null;
let textFewShotExamples = [];
// let imageFewShotExamples = []; // No longer needed

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log("Application initialized");
    initializeUI();
    initializeTabs();
    initializeNestedTabs();
    initializeImageUpload();
});

// UI Initialization
function initializeUI() {
    // Add file input listener
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Add animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-8px)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });
}

// Demo and Pricing Functions
function startDemo() {
    const mainContent = document.querySelector('.main-content');
    mainContent.scrollIntoView({ behavior: 'smooth' });
}

function showPricing() {
    alert('Contact us for pricing information!');
}

// File Handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        previewFile(file);
    }
}

function previewFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const content = e.target.result;
            let preview = '';

            if (file.name.endsWith('.json')) {
                const json = JSON.parse(content);
                preview = JSON.stringify(json, null, 2);
            } else {
                preview = content;
            }

            // Show first 500 characters
            if (preview.length > 500) {
                preview = preview.substring(0, 500) + '\n... (truncated)';
            }

            const previewElement = document.getElementById('dataset-content');
            previewElement.textContent = preview;
            previewElement.style.opacity = '0';
            setTimeout(() => {
                previewElement.style.opacity = '1';
            }, 100);
        } catch (error) {
            showError('Error reading file: ' + error.message);
        }
    };
    reader.readAsText(file);
}

// File Upload
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const responseDiv = document.getElementById('uploadResponse');

    if (!file) {
        showError('Please select a file first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        showMessage('Uploading...', 'info');
        const response = await fetch('/upload-text', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            showMessage(data.message, 'success');
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        showError('Error: ' + error.message);
    }
}

// Training Functionality
async function startTextTraining() {
    const progressBar = document.getElementById('progress-bar');
    const progressLabel = document.getElementById('progress-label');
    const trainButton = document.querySelector('button[onclick="startTextTraining()"]');

    try {
        // Disable button and show loading state
        setButtonLoading(trainButton, true);

        const num = parseInt(document.getElementById('numTextClasses').value);
const classNames = [];

for (let i = 0; i < num; i++) {
    const name = document.getElementById(`textClassName-${i}`).value.trim();
    if (name) classNames.push(name);
}

const response = await fetch('/train-text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_classes: classNames.length, class_names: classNames })
});

        if (!response.ok) {
            throw new Error('Training failed to start');
        }

        // Start progress polling
        await pollProgress(progressBar, progressLabel, trainButton);
    } catch (error) {
        console.error('Training error:', error);
        showError('Training error: ' + error.message);
        setButtonLoading(trainButton, false);
    }
}

// Progress Polling
async function pollProgress(progressBar, progressLabel, trainButton) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/progress');
            const data = await response.json();
            const percent = data.percent;

            // Animate progress bar
            progressBar.style.width = percent + '%';
            progressLabel.textContent = percent + '%';

            if (percent >= 100) {
                clearInterval(pollInterval);
                setButtonLoading(trainButton, false);
                showMessage('Training Complete!', 'success');
            }
        } catch (error) {
            console.error('Progress polling error:', error);
            clearInterval(pollInterval);
            showError('Error checking progress');
            setButtonLoading(trainButton, false);
        }
    }, 1000);
}

// Inference
async function runInference() {
    const userInput = document.getElementById('userInput').value;
    const resultDiv = document.getElementById('inferenceResult');
    const classifyButton = document.querySelector('button[onclick="runInference()"]');

    if (!userInput.trim()) {
        showError('Please enter some text to classify.');
        return;
    }

    try {
        setButtonLoading(classifyButton, true);
        resultDiv.textContent = 'Classifying...';
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: userInput })
        });

        const data = await response.json();
        
        if (response.ok) {
            showResult(data.result);
        } else {
            throw new Error(data.error || 'Classification failed');
        }
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        setButtonLoading(classifyButton, false);
    }
}

// UI Helpers
function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.innerHTML = '<span class="loading-spinner"></span> Processing...';
    } else {
        button.disabled = false;
        button.textContent = button.getAttribute('data-original-text') || 'Submit';
    }
}

function showMessage(message, type = 'info') {
    const responseDiv = document.getElementById('uploadResponse');
    responseDiv.textContent = message;
    responseDiv.className = 'response-message ' + type;
    responseDiv.style.opacity = '0';
    setTimeout(() => {
        responseDiv.style.opacity = '1';
    }, 100);
}

function showError(message) {
    showMessage(message, 'error');
}

function showResult(result) {
    const resultDiv = document.getElementById('inferenceResult');
    resultDiv.textContent = 'Classification: ' + result;
    resultDiv.className = 'result-box success';
    resultDiv.style.opacity = '0';
    setTimeout(() => {
        resultDiv.style.opacity = '1';
    }, 100);
}

function generateTextClassInputs() {
    const num = parseInt(document.getElementById('numTextClasses').value);
    const section = document.getElementById('textClassInputs');
    section.innerHTML = "";

    if (isNaN(num) || num < 1) {
        section.innerHTML = "Please enter a valid number of classes.";
        return;
    }

    for (let i = 0; i < num; i++) {
        const div = document.createElement("div");
        div.className = 'input-group';
        div.innerHTML = `
            <label for="textClassName-${i}">Class ${i + 1} Name:</label>
            <input type="text" id="textClassName-${i}" placeholder="e.g., Urgent, Casual, Promotional..." />
        `;
        section.appendChild(div);
    }
}

// Upload the JSON file for text data
async function uploadTextFile() {
  const fileInput = document.getElementById('fileInput');
  const uploadResponse = document.getElementById('uploadResponse');

  if (!fileInput.files[0]) {
    uploadResponse.innerText = "Please select a file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch('/upload-text', {
    method: 'POST',
    body: formData
  });

  const data = await res.json();
  uploadResponse.innerText = data.message || "Upload failed.";
}

// Upload image folder for training
async function uploadImageFolder() {
  const files = document.getElementById('imageFolder').files;
  const imageUploadResponse = document.getElementById('imageUploadResponse');

  if (!files.length) {
    imageUploadResponse.innerText = "Please select image folder.";
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("images", file);
  }

  const res = await fetch('/upload-images', {
    method: 'POST',
    body: formData
  });

  const data = await res.json();
  imageUploadResponse.innerText = data.message || "Image upload failed.";
}

// Start text model training
async function startTextTraining() {
    try {
        resetProgress();
        console.log("Starting text training...");

        // Collect class names from UI
        const num = parseInt(document.getElementById('numTextClasses').value);
        const classNames = [];

        for (let i = 0; i < num; i++) {
            const name = document.getElementById(`textClassName-${i}`).value.trim();
            if (name) classNames.push(name);
        }

        // Send class count + class names
        const response = await fetch('/train-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_classes: classNames.length,
                class_names: classNames
            })
        });

        if (!response.ok) {
            throw new Error(`Training failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log("Training started:", data);
        pollProgress();
    } catch (error) {
        console.error("Training error:", error);
        alert(`Training failed: ${error.message}`);
    }
}

// Start image model training
async function startImageTraining() {
  try {
    resetProgress();
    console.log("Starting image training...");
    const response = await fetch('/train-image', { method: 'POST' });
    
    if (!response.ok) {
      throw new Error(`Training failed: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log("Training started:", data);
    pollProgress();
  } catch (error) {
    console.error("Training error:", error);
    alert(`Training failed: ${error.message}`);
  }
}

// Reset progress bar
function resetProgress() {
  document.getElementById('progress-bar').style.width = "0%";
  document.getElementById('progress-label').innerText = "0%";
}

// Poll progress bar
async function pollProgress() {
  const res = await fetch('/progress');
  const data = await res.json();
  const percent = data.percent;

  document.getElementById('progress-bar').style.width = percent + "%";
  document.getElementById('progress-label').innerText = percent + "%";

  if (percent < 100) {
    setTimeout(pollProgress, 1000); // keep polling
  } else {
    alert("Training complete!");
  }
}

// Run inference on text input
async function runInference() {
  const text = document.getElementById("userInput").value;
  const resultEl = document.getElementById("inferenceResult");

  if (!text) {
    resultEl.innerText = "Please enter some text.";
    return;
  }

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  const data = await res.json();
  resultEl.innerText = `Label: ${data.label} (Confidence: ${data.confidence}%)`;
}

function generateClassInputs() {
  const num = parseInt(document.getElementById('numClasses').value);
  const section = document.getElementById('classCaptureSection');
  section.innerHTML = "";

  if (isNaN(num) || num < 1) {
    section.innerHTML = "Please enter a valid number of classes.";
    return;
  }

  for (let i = 0; i < num; i++) {
    const div = document.createElement("div");
    div.innerHTML = `
      <input type="text" placeholder="Class ${i + 1} name" id="class-${i}" />
      <button onclick="captureAndUpload(${i})">Capture Image</button>
    `;
    section.appendChild(div);
  }

  startWebcam();
}

async function startWebcam() {
  const webcam = document.getElementById("webcam");
  if (navigator.mediaDevices.getUserMedia) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      webcam.srcObject = stream;
    } catch (err) {
      alert("Webcam access denied or unavailable.");
    }
  }
}

async function captureAndUpload(index) {
  const className = document.getElementById(`class-${index}`).value.trim();
  if (!className) {
    alert("Please enter a class name.");
    return;
  }

  const video = document.getElementById("webcam");
  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async function (blob) {
    const formData = new FormData();
    formData.append("file", blob, "frame.png");
    formData.append("class_name", className);

    const res = await fetch("/upload-webcam", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    alert(data.message || "Upload failed.");
  }, "image/png");
}

// Tab functionality
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab-button');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Deactivate all tabs and content
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Activate the clicked tab and its content
            tab.classList.add('active');
            const tabContentId = tab.dataset.tab + '-section';
            document.getElementById(tabContentId).classList.add('active');
        });
    });
}

function initializeNestedTabs() {
    const nestedTabButtons = document.querySelectorAll('.tab-button-nested');

    nestedTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const parentCard = button.closest('.card');
            const tabGroupName = button.dataset.tab.split('-')[0]; // 'text' or 'image'

            // Deactivate sibling buttons and content
            parentCard.querySelectorAll('.tab-button-nested').forEach(btn => {
                if (btn.dataset.tab.startsWith(tabGroupName)) {
                    btn.classList.remove('active');
                }
            });
            parentCard.querySelectorAll('.tab-content-nested').forEach(content => {
                if (content.id.startsWith(tabGroupName)) {
                    content.classList.remove('active');
                }
            });

            // Activate clicked button and its content
            button.classList.add('active');
            const contentId = button.dataset.tab + '-section';
            document.getElementById(contentId).classList.add('active');
        });
    });
}

// Webcam functionality
async function initializeWebcam() {
    if (!cameraStreams.main) {  // Only initialize if not already running
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            
            cameraStreams.main = stream;
            
            const videos = ['webcam', 'webcam-test'];
            videos.forEach(id => {
                const video = document.getElementById(id);
                if (video) {
                    video.srcObject = stream;
                }
            });
        } catch (error) {
            console.error('Error accessing webcam:', error);
            showError('Unable to access webcam. Please ensure camera permissions are granted.');
        }
    }
}

function stopWebcam() {
    // Stop main stream if it exists
    if (cameraStreams.main) {
        cameraStreams.main.getTracks().forEach(track => track.stop());
        delete cameraStreams.main;

        // Clear video sources
        const videos = ['webcam', 'webcam-test'];
        videos.forEach(id => {
            const video = document.getElementById(id);
            if (video) {
                video.srcObject = null;
            }
        });

        // Stop live classification if running
        if (isLiveClassification) {
            toggleLiveClassification();
        }
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    // Stop all camera streams
    Object.values(cameraStreams).forEach(stream => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    // Clear all intervals
    if (classificationInterval) {
        clearInterval(classificationInterval);
    }
});

function switchImageTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.image-tab-button').forEach(button => {
        button.classList.toggle('active', button.dataset.imageTab === tab);
    });

    // Update tab content
    document.querySelectorAll('.image-tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tab}-section`);
    });

    // Handle webcam - stop first to ensure cleanup
    stopWebcam();
    
    // Only initialize if switching to capture tab
    if (tab === 'capture') {
        initializeWebcam();
    }
}

// Class input generation
function generateClassInputs() {
    const numClasses = parseInt(document.getElementById('numClasses').value);
    const container = document.getElementById('classCaptureSection');
    container.innerHTML = '';

    for (let i = 0; i < numClasses; i++) {
        const group = document.createElement('div');
        group.className = 'class-input-group';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = `Class ${i + 1} name`;
        input.id = `class${i}`;
        
        const captureBtn = document.createElement('button');
        captureBtn.className = 'button-secondary';
        captureBtn.textContent = 'Capture';
        captureBtn.onclick = () => captureImage(i);
        
        const count = document.createElement('span');
        count.className = 'capture-count';
        count.id = `count${i}`;
        count.textContent = '0 images';
        
        group.appendChild(input);
        group.appendChild(captureBtn);
        group.appendChild(count);
        container.appendChild(group);
    }
}

// Image capture functionality
const capturedImages = {};

function captureImage(classIndex) {
    const className = document.getElementById(`class${classIndex}`).value;
    if (!className) {
        showError('Please enter a class name first');
        return;
    }

    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    // Store image
    if (!capturedImages[className]) {
        capturedImages[className] = [];
    }
    capturedImages[className].push(imageData);

    // Update count
    const countElement = document.getElementById(`count${classIndex}`);
    countElement.textContent = `${capturedImages[className].length} images`;

    // Show success message
    showMessage(`Captured image for class "${className}"`, 'success');
}

// Image training functionality
async function startImageTraining() {
    const trainButton = document.querySelector('button[onclick="startImageTraining()"]');
    const progressBar = document.getElementById('image-progress-bar');
    const progressLabel = document.getElementById('image-progress-label');

    try {
        // Check if we have images from either method
        const hasUploadedImages = uploadedImages.size > 0;
        const hasCapturedImages = Object.keys(capturedImages).length > 0;

        if (!hasUploadedImages && !hasCapturedImages) {
            throw new Error('Please add some images first (either upload or capture)');
        }

        setButtonLoading(trainButton, true);

        // Prepare form data with all images
        const formData = new FormData();

        // Add uploaded images
        uploadedImages.forEach((image, id) => {
            if (image.class) {
                const blob = dataURLtoBlob(image.data);
                formData.append('images', blob, `${image.class}_${id}.jpg`);
            }
        });

        // Add captured images
        Object.entries(capturedImages).forEach(([className, images]) => {
            images.forEach((imageData, index) => {
                const blob = dataURLtoBlob(imageData);
                formData.append('images', blob, `${className}_${index}.jpg`);
            });
        });

        // Upload images
        const uploadResponse = await fetch('/upload-images', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Failed to upload images');
        }

        // Start training
        const trainResponse = await fetch('/train-image', {
            method: 'POST'
        });

        if (!trainResponse.ok) {
            throw new Error('Failed to start training');
        }

        // Poll progress
        await pollImageProgress(progressBar, progressLabel, trainButton);
    } catch (error) {
        console.error('Training error:', error);
        showError(error.message);
        setButtonLoading(trainButton, false);
    }
}

// Image progress polling
async function pollImageProgress(progressBar, progressLabel, trainButton) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/progress');
            const data = await response.json();
            const percent = data.percent;

            progressBar.style.width = percent + '%';
            progressLabel.textContent = percent + '%';

            if (percent >= 100) {
                clearInterval(pollInterval);
                setButtonLoading(trainButton, false);
                showMessage('Image training complete!', 'success');
            }
        } catch (error) {
            console.error('Progress polling error:', error);
            clearInterval(pollInterval);
            showError('Error checking progress');
            setButtonLoading(trainButton, false);
        }
    }, 1000);
}

// Image classification functionality
async function captureAndClassify() {
    const video = document.getElementById('webcam-test');
    const canvas = document.getElementById('canvas-test');
    const context = canvas.getContext('2d');
    const resultDiv = document.getElementById('imageInferenceResult');

    try {
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get image data
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Send for classification
        const response = await fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                task: 'image',
                input: imageData
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            showResult(data.result, 'imageInferenceResult');
        } else {
            throw new Error(data.error || 'Classification failed');
        }
    } catch (error) {
        showError(error.message);
    }
}

// Live classification
function toggleLiveClassification() {
    const button = document.querySelector('button[onclick="toggleLiveClassification()"]');
    
    if (!isLiveClassification) {
        // Start live classification
        isLiveClassification = true;
        button.textContent = 'Stop Live Classification';
        button.classList.add('active');
        
        classificationInterval = setInterval(captureAndClassify, 1000);
    } else {
        // Stop live classification
        isLiveClassification = false;
        button.textContent = 'Start Live Classification';
        button.classList.remove('active');
        
        if (classificationInterval) {
            clearInterval(classificationInterval);
            classificationInterval = null;
        }
    }
}

// Utility functions
function dataURLtoBlob(dataURL) {
    const arr = dataURL.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    
    return new Blob([u8arr], { type: mime });
}

// Image upload functionality
const uploadedImages = new Map(); // Store uploaded images with their class assignments

function initializeImageUpload() {
    const dropzone = document.getElementById('image-dropzone');
    const fileInput = document.getElementById('imageInput');

    // File input change handler
    fileInput.addEventListener('change', handleImageSelect);

    // Drag and drop handlers
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleImageDrop(e.dataTransfer.files);
    });

    // Click to upload
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // Initialize image tabs
    initializeImageTabs();
}

function initializeImageTabs() {
    const tabButtons = document.querySelectorAll('.image-tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tab = button.dataset.imageTab;
            switchImageTab(tab);
        });
    });
}

function handleImageSelect(event) {
    const files = event.target.files;
    handleImageFiles(files);
}

function handleImageDrop(files) {
    handleImageFiles(files);
}

function handleImageFiles(files) {
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    imageFiles.forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const imageId = Date.now() + Math.random().toString(36).substr(2, 9);
            uploadedImages.set(imageId, {
                data: e.target.result,
                class: null,
                file: file
            });
            updateImageGrid();
        };
        reader.readAsDataURL(file);
    });
}

function updateImageGrid() {
    const grid = document.getElementById('image-preview-grid');
    grid.innerHTML = '';

    uploadedImages.forEach((image, id) => {
        const item = document.createElement('div');
        item.className = 'image-item';
        
        const img = document.createElement('img');
        img.src = image.data;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-button';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            uploadedImages.delete(id);
            updateImageGrid();
        };

        if (image.class) {
            const label = document.createElement('div');
            label.className = 'class-label';
            label.textContent = image.class;
            item.appendChild(label);
        }

        item.appendChild(img);
        item.appendChild(removeBtn);
        grid.appendChild(item);
    });
}

function assignClass() {
    const className = document.getElementById('uploadClassName').value.trim();
    if (!className) {
        showError('Please enter a class name');
        return;
    }

    let assigned = false;
    uploadedImages.forEach((image, id) => {
        if (!image.class) {
            image.class = className;
            assigned = true;
        }
    });

    if (assigned) {
        updateImageGrid();
        showMessage(`Images assigned to class "${className}"`, 'success');
    } else {
        showMessage('No unassigned images found', 'info');
    }
}

function addTextExample() {
    const textInput = document.getElementById('text-few-shot-input');
    const classInput = document.getElementById('text-few-shot-class');
    
    if (textInput.value.trim() && classInput.value.trim()) {
        textFewShotExamples.push({
            text: textInput.value,
            "class": classInput.value
        });
        updateTextExamplesUI();
        textInput.value = "";
        classInput.value = "";
    } else {
        alert("Please provide both example text and a class.");
    }
}

function updateTextExamplesUI() {
    const container = document.getElementById('text-few-shot-examples');
    container.innerHTML = '<h4>Collected Examples:</h4>';
    textFewShotExamples.forEach(ex => {
        const p = document.createElement('p');
        p.textContent = `"${ex.text}" -> ${ex.class}`;
        container.appendChild(p);
    });
}

async function classifyTextFewShot() {
    const textToClassify = document.getElementById('text-classify-input').value;
    const resultDiv = document.getElementById('text-few-shot-result');

    if (!textToClassify.trim()) {
        alert("Please enter text to classify.");
        return;
    }

    if (textFewShotExamples.length === 0) {
        alert("Please add at least one few-shot example.");
        return;
    }

    resultDiv.textContent = 'Classifying...';
    const response = await fetch('/few-shot-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            examples: textFewShotExamples,
            text: textToClassify
        })
    });
    const data = await response.json();
    resultDiv.textContent = `Result: ${data.result}`;
}

async function classifyImageFewShot() {
    const fileInput = document.getElementById('imageClassifyInput');
    const file = fileInput.files[0];
    const resultDiv = document.getElementById('image-few-shot-result');

    if (!file) {
        alert("Please select an image to classify.");
        return;
    }

    const formData = new FormData();
    formData.append('image', file);
    
    resultDiv.textContent = 'Classifying...';
    const response = await fetch('/few-shot-image', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    resultDiv.textContent = `Result: ${data.result}`;
}

async function uploadFewShotFile() {
    const fileInput = document.getElementById('fewShotFileInput');
    const file = fileInput.files[0];
    const responseDiv = document.getElementById('fewShotUploadResponse');

    if (!file) {
        responseDiv.textContent = 'Please select a file.';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        responseDiv.textContent = 'Uploading...';

        const res = await fetch('/upload-few-shot-text', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (res.ok) {
            responseDiv.textContent = data.message || 'Upload successful!';
            if (data.examples) {
                textFewShotExamples = data.examples;
                updateTextExamplesUI();
            }
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (err) {
        responseDiv.textContent = 'Error: ' + err.message;
    }
}

async function classifyTextFewShot() {
    const textToClassify = document.getElementById('text-classify-input').value;
    const resultDiv = document.getElementById('text-few-shot-result');
    const taskPrompt = document.getElementById('taskDescription').value.trim();

    if (!textToClassify) {
        alert("Please enter text to classify.");
        return;
    }

    if (textFewShotExamples.length === 0) {
        alert("Please upload a few-shot dataset first.");
        return;
    }

    resultDiv.textContent = 'Classifying...';

    const response = await fetch('/few-shot-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            task: taskPrompt,
            examples: textFewShotExamples,
            text: textToClassify
        })
    });

    const data = await response.json();
    resultDiv.textContent = `Result: ${data.result}`;
}

async function submitTaskPrompt() {
    const taskPrompt = document.getElementById("taskDescription").value.trim();
    const responseDiv = document.getElementById("taskPromptResponse");

    if (!taskPrompt) {
        responseDiv.textContent = "Please enter a task description.";
        responseDiv.className = "response-message error";
        return;
    }

    try {
        responseDiv.textContent = "Submitting...";
        const response = await fetch('/submit-task', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task: taskPrompt })
        });

        const data = await response.json();
        if (response.ok) {
            responseDiv.textContent = data.message || "Task submitted successfully.";
            responseDiv.className = "response-message success";
        } else {
            throw new Error(data.error || "Add the dataset first");
        }
    } catch (err) {
        responseDiv.textContent = "Error: " + err.message;
        responseDiv.className = "response-message error";
    }
}

function generateImageClassDropzones() {
    const num = parseInt(document.getElementById('numImageClasses').value);
    const container = document.getElementById('multiClassDropzones');
    container.innerHTML = '';

    if (isNaN(num) || num < 1) {
        container.innerHTML = '<p class="error-message">Please enter a valid number of classes.</p>';
        return;
    }

    for (let i = 0; i < num; i++) {
        const classId = `class-${i}`;
        const dropDiv = document.createElement('div');
        dropDiv.className = 'dropzone-class-wrapper';
        dropDiv.innerHTML = `
            <div class="input-group">
                <label for="${classId}-name">Class ${i + 1} Name</label>
                <input type="text" id="${classId}-name" placeholder="e.g., Dog, Cat..." />
            </div>
            <div class="dropzone" id="${classId}-dropzone">
                <div class="dropzone-content">
                    <span class="upload-icon">📁</span>
                    <p>Drag & drop images or click to upload</p>
                    <input type="file" multiple accept="image/*" class="file-input" />
                </div>
            </div>
            <div id="${classId}-preview" class="image-grid"></div>
        `;
        container.appendChild(dropDiv);

        const dropzone = dropDiv.querySelector('.dropzone');
        const fileInput = dropzone.querySelector('.file-input');
        
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleClassImageUpload(e.dataTransfer.files, classId);
        });
        fileInput.addEventListener('change', (e) => handleClassImageUpload(e.target.files, classId));
    }
}

const classImageMap = new Map();
let globalImageUploadTimestamp = null;

async function handleClassImageUpload(files, classId) {
    // ── Guard clauses ────────────────────────────────────────────
    if (!files || files.length === 0) return;

    const classNameInput = document.getElementById(`${classId}-name`);
    const className      = classNameInput ? classNameInput.value.trim() : null;

    if (!className) {
        showError("Please enter a class name before uploading.");
        return;
    }

    const previewContainer = document.getElementById(`${classId}-preview`);
    if (!previewContainer) return;

    // ── Shared timestamp for this upload session ─────────────────
    if (!globalImageUploadTimestamp) {
        globalImageUploadTimestamp = new Date()
            .toISOString()        // e.g. 2024-06-22T14:23:55.123Z
            .replace(/[-T:.Z]/g, "") // strip non-digits → 20240622142355123
            .slice(0, 14);           // keep yyyymmddHHMMSS
    }

    // ── Show "Uploading…" placeholder ────────────────────────────
    const loadingMsg = document.createElement("div");
    loadingMsg.className = "loading-message";
    loadingMsg.textContent = "Uploading images…";
    previewContainer.appendChild(loadingMsg);

    // ── Build multipart form data ────────────────────────────────
    const formData = new FormData();
    formData.append("class_name", className);
    formData.append("timestamp_dir", globalImageUploadTimestamp);

    Array.from(files).forEach(file => {
        if (file.type.startsWith("image/")) {
            formData.append("files", file);
        }
    });

    try {
        const response = await fetch("/upload-images", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        // Remove loader
        previewContainer.removeChild(loadingMsg);

        if (!response.ok) {
            throw new Error(data.error || "Upload failed");
        }

        // ── Success message & thumbnails ────────────────────────
        showMessage(
            `${data.message} (${data.total_files} files saved)`,
            "success"
        );

        Array.from(files).forEach(file => {
            if (!file.type.startsWith("image/")) return;

            const reader = new FileReader();
            reader.onload = e => {
                const imgContainer = document.createElement("div");
                imgContainer.className = "preview-image-container";

                const img   = document.createElement("img");
                img.src     = e.target.result;
                img.className = "preview-image";

                const label = document.createElement("div");
                label.className = "image-label";
                label.textContent = className;

                imgContainer.appendChild(img);
                imgContainer.appendChild(label);
                previewContainer.appendChild(imgContainer);
            };
            reader.readAsDataURL(file);
        });

        console.log("Upload successful:", data);
    } catch (error) {
        if (previewContainer.contains(loadingMsg)) {
            previewContainer.removeChild(loadingMsg);
        }
        showError("Error uploading images: " + error.message);
        console.error("Upload error:", error);
    }
}

// Also add this helper function to show better error/success messages
function showMessage(message, type = 'info') {
    // Create or update a message container
    let messageContainer = document.getElementById('message-container');
    if (!messageContainer) {
        messageContainer = document.createElement('div');
        messageContainer.id = 'message-container';
        messageContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 300px;
        `;
        document.body.appendChild(messageContainer);
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.style.cssText = `
        padding: 12px;
        margin: 5px 0;
        border-radius: 4px;
        color: white;
        background-color: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    messageDiv.textContent = message;

    messageContainer.appendChild(messageDiv);

    // Fade in
    setTimeout(() => {
        messageDiv.style.opacity = '1';
    }, 100);

    // Remove after 5 seconds
    setTimeout(() => {
        messageDiv.style.opacity = '0';
        setTimeout(() => {
            if (messageContainer.contains(messageDiv)) {
                messageContainer.removeChild(messageDiv);
            }
        }, 300);
    }, 5000);
}

// Add this CSS to your HTML file for better styling
const additionalCSS = `
.preview-image-container {
    position: relative;
    display: inline-block;
    margin: 5px;
}

.preview-image {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 4px;
    border: 2px solid #ddd;
}

.image-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 2px 4px;
    font-size: 12px;
    text-align: center;
    border-radius: 0 0 4px 4px;
}

.loading-message {
    padding: 10px;
    background-color: #f0f0f0;
    border-radius: 4px;
    text-align: center;
    margin: 10px 0;
}

.message {
    font-size: 14px;
    font-weight: 500;
}
`;

// Add the CSS to the page
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalCSS;
document.head.appendChild(styleSheet);


// Initialize webcam when 'Real-time Capture' tab is clicked
document.querySelectorAll('.image-tab-button').forEach(button => {
    button.addEventListener('click', () => {
        const target = button.getAttribute('data-image-tab');

        // Tab switching logic
        document.querySelectorAll('.image-tab-button').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        document.querySelectorAll('.image-tab-content').forEach(tab => tab.classList.remove('active'));
        document.getElementById(`${target}-section`).classList.add('active');

        // If it's the capture tab, activate webcam
        if (target === 'capture') {
            activateWebcam();
        }
    });
});

function activateWebcam() {
    const video = document.getElementById('webcam');
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.play();
            })
            .catch(error => {
                console.error("Webcam access denied or not available", error);
            });
    }
}


// Global variables for real-time classification
let realtimeClassificationActive = false;
let classificationIntervalId = null;
let lastClassificationTime = 0;
const CLASSIFICATION_DELAY = 500; // 500ms between classifications

// Enhanced webcam initialization with better error handling
async function initializeWebcamEnhanced(videoId = 'webcam-test') {
    try {
        const video = document.getElementById(videoId);
        if (!video) {
            throw new Error(`Video element '${videoId}' not found`);
        }

        // Stop any existing stream
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }

        // Request webcam access with preferred settings
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640, max: 1280 },
                height: { ideal: 480, max: 720 },
                frameRate: { ideal: 30, max: 60 }
            }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            console.log('✅ Webcam initialized successfully');
        };

        return true;
    } catch (error) {
        console.error('❌ Webcam initialization failed:', error);
        showError(`Webcam error: ${error.message}`);
        return false;
    }
}

// Enhanced capture and classify with better performance
async function captureAndClassifyEnhanced() {
    const video = document.getElementById('webcam-test');
    const canvas = document.getElementById('canvas-test');
    const resultDiv = document.getElementById('imageInferenceResult');
    const classifyButton = document.querySelector('button[onclick="captureAndClassifyEnhanced()"]');

    if (!video || !canvas) {
        showError('Video or canvas element not found');
        return;
    }

    try {
        setButtonLoading(classifyButton, true);
        resultDiv.textContent = 'Capturing and classifying...';

        // Ensure webcam is initialized before capture
        if (video && !video.srcObject) {
            const ready = await initializeWebcamEnhanced('webcam-test');
            if (!ready) {
                showError('Unable to access webcam');
                setButtonLoading(classifyButton, false);
                return;
            }
            // Give the webcam a brief moment to start streaming
            await new Promise(r => setTimeout(r, 300));
        }

        // Capture frame
        const imageData = captureVideoFrame(video, canvas);
        
        // Send for classification
        const result = await classifyImageData(imageData);
        
        if (result.error) {
            throw new Error(result.error);
        }

        // Display result with enhanced formatting
        displayClassificationResult(result, 'imageInferenceResult');
        
    } catch (error) {
        console.error('Classification error:', error);
        showError(`Classification failed: ${error.message}`);
    } finally {
        setButtonLoading(classifyButton, false);
    }
}

// Capture video frame to base64
function captureVideoFrame(video, canvas) {
    const context = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64 with good quality
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Enhanced image classification function
async function classifyImageData(imageData) {
    try {
        const response = await fetch('/classify-webcam', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        return { error: error.message };
    }
}

// Enhanced real-time classification toggle
async function toggleLiveClassificationEnhanced() {
    const button = document.querySelector('button[onclick="toggleLiveClassificationEnhanced()"]');
    const video = document.getElementById('webcam-test');
    const statusDiv = document.getElementById('imageInferenceResult');

    if (!realtimeClassificationActive) {
        // Start real-time classification
        try {
            // Ensure webcam is running
            const webcamReady = await initializeWebcamEnhanced('webcam-test');
            if (!webcamReady) {
                throw new Error('Failed to initialize webcam');
            }

            realtimeClassificationActive = true;
            button.textContent = '⏹️ Stop Live Classification';
            button.classList.add('active');
            statusDiv.textContent = '🔄 Starting live classification...';

            // Start the classification loop
            startRealtimeClassificationLoop();
            
            console.log('✅ Live classification started');
        } catch (error) {
            console.error('Failed to start live classification:', error);
            showError(`Failed to start live classification: ${error.message}`);
            realtimeClassificationActive = false;
        }
    } else {
        // Stop real-time classification
        stopRealtimeClassification();
        button.textContent = '▶️ Start Live Classification';
        button.classList.remove('active');
        statusDiv.textContent = '⏸️ Live classification stopped';
        console.log('⏹️ Live classification stopped');
    }
}

// Real-time classification loop
function startRealtimeClassificationLoop() {
    const video = document.getElementById('webcam-test');
    const canvas = document.getElementById('canvas-test');
    
    const classifyLoop = async () => {
        if (!realtimeClassificationActive) return;

        const currentTime = Date.now();
        
        // Throttle classifications to avoid overwhelming the server
        if (currentTime - lastClassificationTime >= CLASSIFICATION_DELAY) {
            try {
                const imageData = captureVideoFrame(video, canvas);
                const result = await classifyImageData(imageData);
                
                if (!result.error) {
                    displayClassificationResult(result, 'imageInferenceResult', true);
                }
                
                lastClassificationTime = currentTime;
            } catch (error) {
                console.error('Real-time classification error:', error);
                // Don't show errors for every frame to avoid spam
            }
        }

        // Schedule next classification
        if (realtimeClassificationActive) {
            classificationIntervalId = setTimeout(classifyLoop, 100); // Check every 100ms
        }
    };

    // Start the loop
    classifyLoop();
}

// Stop real-time classification
function stopRealtimeClassification() {
    realtimeClassificationActive = false;
    if (classificationIntervalId) {
        clearTimeout(classificationIntervalId);
        classificationIntervalId = null;
    }
}

// Enhanced result display
function displayClassificationResult(result, elementId, isRealtime = false) {
    const resultDiv = document.getElementById(elementId);
    if (!resultDiv) return;

    const timestamp = new Date().toLocaleTimeString();
    const confidence = result.confidence || 0;
    const label = result.label || 'Unknown';
    
    // Create enhanced result display
    const resultHTML = `
        <div class="classification-result ${isRealtime ? 'realtime' : 'single'}">
            <div class="result-header">
                <span class="result-label">${label}</span>
                <span class="result-confidence">${confidence.toFixed(1)}%</span>
            </div>
            <div class="result-details">
                <span class="result-timestamp">${timestamp}</span>
                ${isRealtime ? '<span class="realtime-indicator">🔴 Live</span>' : ''}
            </div>
            ${confidence > 80 ? '<div class="confidence-bar high"></div>' : 
              confidence > 60 ? '<div class="confidence-bar medium"></div>' : 
              '<div class="confidence-bar low"></div>'}
        </div>
    `;
    
    resultDiv.innerHTML = resultHTML;
    resultDiv.className = `result-box ${confidence > 70 ? 'high-confidence' : 'low-confidence'}`;
}