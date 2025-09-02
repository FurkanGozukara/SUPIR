// --- START OF FILE compare_fullscreen.js ---

// Global state for zoom and pan - potentially one set per active slider?
// For simplicity, assume only one slider is actively zoomed/panned at a time.
let zoomLevel = 1;
let offsetX = 0;
let offsetY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let imageWrapper = null; // Element being transformed (contains images)
let sliderElement = null; // The noUiSlider DOM element being interacted with
let clipElement = null; // The element whose width creates the compare effect
let currentSliderId = null; // Keep track of which slider is active

const MIN_ZOOM = 1;
const MAX_ZOOM = 10;
const ZOOM_SENSITIVITY = 0.005;

// Function to apply the current transform (zoom and pan) to the image wrapper
function applyTransform() {
    // Check if the *currently tracked* wrapper exists
    if (!imageWrapper || !sliderElement) {
        // console.warn("ApplyTransform: Missing imageWrapper or sliderElement for slider ID:", currentSliderId);
        return;
    }

    const container = imageWrapper.parentElement;
    if (!container) return;
    const containerRect = container.getBoundingClientRect();
    // Find the first *visible* image for dimensions (important for slider with potentially hidden images)
    const firstImage = imageWrapper.querySelector('img:not([style*="display: none"])');
    if (!firstImage) {
        // Fallback if no non-hidden image found (less accurate)
        const anyImage = imageWrapper.querySelector('img');
        if (!anyImage) {
            // console.warn("ApplyTransform: No image found in wrapper for slider ID:", currentSliderId);
            return;
        }
        console.warn("ApplyTransform: Using potentially hidden image for size reference.");
        firstImage = anyImage;
    }


    // Estimate content size (can be refined if images differ significantly)
    // Use displayed size if available, else natural size
    const currentImgWidth = firstImage.clientWidth || firstImage.naturalWidth;
    const currentImgHeight = firstImage.clientHeight || firstImage.naturalHeight;

    // If dimensions are still 0, we can't proceed
    if (!currentImgWidth || !currentImgHeight) {
        console.warn("ApplyTransform: Image dimensions are zero for slider ID:", currentSliderId, firstImage);
        // Attempt to reset might be safest here if this happens unexpectedly
        // resetZoomPan(true, currentSliderId); // Pass ID if resetting
        return;
    }

    const contentWidth = currentImgWidth * zoomLevel;
    const contentHeight = currentImgHeight * zoomLevel;

    // Calculate max allowable offsets relative to the container size
    const maxOverflowX = Math.max(0, contentWidth - containerRect.width);
    const maxOverflowY = Math.max(0, contentHeight - containerRect.height);

    const maxOffsetX = maxOverflowX / 2;
    const minOffsetX = -maxOverflowX / 2;
    const maxOffsetY = maxOverflowY / 2;
    const minOffsetY = -maxOverflowY / 2;

    // Clamp current offsets
    offsetX = Math.max(minOffsetX, Math.min(maxOffsetX, offsetX));
    offsetY = Math.max(minOffsetY, Math.min(maxOffsetY, offsetY));

    // Apply the transform
    imageWrapper.style.transformOrigin = 'center center';
    imageWrapper.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${zoomLevel})`;

    // Trigger manual update of the slider clipping
    updateClipElementWidth(); // Uses the globally tracked sliderElement
}

// Function to reset zoom and pan for the current slider
function resetZoomPan(apply = true, sliderIdToReset = null) {
    const targetSliderId = sliderIdToReset || currentSliderId;
    console.log(`Resetting zoom and pan for slider: ${targetSliderId}`);

    // Use the globally tracked elements if resetting the current one
    // otherwise, find them if a specific different sliderId is provided
    let wrapperToReset = imageWrapper;
    let sliderElemToReset = sliderElement;
    let clipElemToReset = clipElement;

    if (sliderIdToReset && sliderIdToReset !== currentSliderId) {
        console.log("Resetting elements for a non-current slider:", sliderIdToReset);
        sliderElemToReset = document.getElementById(sliderIdToReset);
        if (sliderElemToReset) {
            wrapperToReset = sliderElemToReset.querySelector('.image-zoom-wrapper') || sliderElemToReset.querySelector('.slider-wrap') || sliderElemToReset.querySelector('.noUi-base');
            clipElemToReset = sliderElemToReset.querySelector('.noUi-connect');
        } else {
            console.error("Cannot find slider element to reset:", sliderIdToReset);
            return; // Exit if slider not found
        }
    } else if (!targetSliderId) {
        console.warn("ResetZoomPan called with no targetSliderId and no currentSliderId.");
        return;
    }


    // Reset state variables *first*
    zoomLevel = 1;
    offsetX = 0;
    offsetY = 0;
    isPanning = false; // Ensure panning stops

    if (wrapperToReset) {
        wrapperToReset.style.cursor = 'default';
        wrapperToReset.style.transform = 'translate(0px, 0px) scale(1)';
        // We need to update clipping based on the *reset* state
        // Ensure the correct slider/clip elements are used for the update call
        const originalSliderElement = sliderElement;
        const originalClipElement = clipElement;
        sliderElement = sliderElemToReset;
        clipElement = clipElemToReset;
        if (apply) {
            applyTransform(); // Apply the reset state + update clipping
        } else {
            updateClipElementWidth(); // Ensure clipping is correct even if transform isn't fully applied yet
        }
        // Restore original elements if they were temporarily changed
        sliderElement = originalSliderElement;
        clipElement = originalClipElement;

    } else {
        console.warn("No image wrapper found during reset for slider:", targetSliderId);
    }

    // If resetting the *current* slider, clear global tracking vars
    if (targetSliderId === currentSliderId) {
        console.log("Clearing global tracking variables as current slider was reset.");
        // Don't null out sliderElement here if applyTransform needs it later?
        // Only null out if truly cleaning up. Let cleanupZoomPan handle that.
        // imageWrapper = null; // Keep reference until cleanup
        // sliderElement = null;
        // clipElement = null;
        // currentSliderId = null; // Keep track until cleanup
    }
}


// Event handler for mouse wheel zoom
function handleWheelZoom(event) {
    // Use the globally tracked imageWrapper
    if (!imageWrapper || isPanning) return;

    event.preventDefault();

    const container = imageWrapper.parentElement;
    if (!container) return;
    const rect = container.getBoundingClientRect();

    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const mouseRelCenterX = mouseX - rect.width / 2;
    const mouseRelCenterY = mouseY - rect.height / 2;

    const delta = event.deltaY * ZOOM_SENSITIVITY;
    const oldZoomLevel = zoomLevel;
    const newZoomLevel = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, oldZoomLevel * (1 - delta)));
    const zoomRatio = newZoomLevel / oldZoomLevel;

    offsetX = offsetX * zoomRatio + mouseRelCenterX * (1 - zoomRatio);
    offsetY = offsetY * zoomRatio + mouseRelCenterY * (1 - zoomRatio);

    zoomLevel = newZoomLevel;

    if (zoomLevel <= MIN_ZOOM) {
        resetZoomPan(true); // Reset the current slider
    } else {
        imageWrapper.style.cursor = 'grab';
        applyTransform();
    }
}


// Event handler for starting panning
function handlePanStart(event) {
    // Use the globally tracked imageWrapper
    if (zoomLevel <= MIN_ZOOM || !imageWrapper) return;
    if (event.button !== 0) return;

    event.preventDefault();
    isPanning = true;
    panStartX = event.clientX - offsetX;
    panStartY = event.clientY - offsetY;
    imageWrapper.style.cursor = 'grabbing';
    imageWrapper.style.userSelect = 'none';

    // Add listeners to document to capture mouse movements anywhere
    document.addEventListener('mousemove', handlePanMove);
    document.addEventListener('mouseup', handlePanEnd);
    document.addEventListener('mouseleave', handlePanEnd);
}

// Event handler for panning movement
function handlePanMove(event) {
    if (!isPanning || !imageWrapper) return;
    event.preventDefault();

    offsetX = event.clientX - panStartX;
    offsetY = event.clientY - panStartY;

    applyTransform(); // Apply transform (includes clamping and clipping update)
}

// Event handler for ending panning
function handlePanEnd(event) {
    if (!isPanning) return;
    if (event.button !== 0 && event.type !== 'mouseleave') return;

    isPanning = false;
    // Use the globally tracked imageWrapper
    if (imageWrapper) {
        imageWrapper.style.cursor = zoomLevel > MIN_ZOOM ? 'grab' : 'default';
        imageWrapper.style.userSelect = '';
    }

    document.removeEventListener('mousemove', handlePanMove);
    document.removeEventListener('mouseup', handlePanEnd);
    document.removeEventListener('mouseleave', handlePanEnd);
}

// Update clipping based on the globally tracked sliderElement and clipElement
function updateClipElementWidth() {
    if (!sliderElement || !sliderElement.noUiSlider || !clipElement || !imageWrapper) {
        // console.warn("Slider/Clip/Wrapper element not ready for clipping update for slider:", currentSliderId);
        return;
    }

    try {
        const sliderInstance = sliderElement.noUiSlider;
        const positions = sliderInstance.getPositions();
        if (positions === undefined || positions.length === 0) {
            console.warn("Could not get slider positions for slider:", currentSliderId);
            return;
        }
        const sliderPercent = positions[0];

        const container = imageWrapper.parentElement;
        if (!container) return;

        const containerRect = container.getBoundingClientRect();
        const wrapperRect = imageWrapper.getBoundingClientRect(); // Get current visual bounds

        // If wrapper has no width/height (e.g., display:none), bail out
        if (wrapperRect.width === 0 || wrapperRect.height === 0) {
            // console.warn("Wrapper has zero dimensions, skipping clip update for slider:", currentSliderId);
            return;
        }

        const targetClipX_inContainer = containerRect.left + (containerRect.width * sliderPercent / 100);
        const targetClipX_inWrapper = targetClipX_inContainer - wrapperRect.left;
        const clipElementWidthPx = Math.max(0, Math.min(wrapperRect.width, targetClipX_inWrapper));

        clipElement.style.width = clipElementWidthPx + 'px';
        clipElement.style.left = '0'; // Ensure it starts from the left edge of its parent

        // Debugging log
        // console.log(`Slider ID: ${currentSliderId} | %: ${sliderPercent.toFixed(1)} | TargetX: ${targetClipX_inContainer.toFixed(1)} | WrapL: ${wrapperRect.left.toFixed(1)} | WrapW: ${wrapperRect.width.toFixed(1)} | ClipW: ${clipElementWidthPx.toFixed(1)}px`);

    } catch (error) {
        console.error("Error updating clip element width for slider:", currentSliderId, error);
    }
}


// Function to initialize zoom/pan listeners for a SPECIFIC slider
function initializeZoomPan(sliderId) {
    console.log(`Initializing zoom and pan for slider: ${sliderId}`);

    // Clean up any existing listeners for the *previous* slider first
    if (currentSliderId && currentSliderId !== sliderId) {
        console.log(`Cleaning up previous slider (${currentSliderId}) before initializing ${sliderId}`);
        cleanupZoomPan(currentSliderId); // Cleanup specific ID
    } else if (currentSliderId === sliderId) {
        console.log(`Re-initializing zoom/pan for the same slider: ${sliderId}. Cleaning up first.`);
        cleanupZoomPan(sliderId); // Cleanup specific ID
    }


    const targetSliderElement = document.getElementById(sliderId);
    if (!targetSliderElement) {
        console.error(`Compare slider with ID '${sliderId}' not found for zoom/pan init.`);
        // Clear global refs if initialization fails
        cleanupZoomPan(null); // Call cleanup with null to reset globals
        return;
    }

    // --- Set global state for the *new* active slider ---
    sliderElement = targetSliderElement;
    currentSliderId = sliderId; // Track the currently active slider

    // Find elements *within the target slider*
    imageWrapper = sliderElement.querySelector('.image-zoom-wrapper');
    if (!imageWrapper) {
        // Check if the slider itself contains the images (common in ImageSlider)
        // The direct children that are image containers (like divs holding the img tags)
        // Need to look inside noUi-base potentially
        const base = sliderElement.querySelector('.noUi-base');
        if (base && base.children.length >= 2 && base.children[0].tagName === 'DIV' && base.children[1].tagName === 'DIV') {
            // Heuristic: If noUi-base has two div children, assume the base itself acts as the wrapper
            // This is fragile and depends heavily on Gradio ImageSlider's internal structure
            imageWrapper = base;
            console.warn(`Using '.noUi-base' as image wrapper for slider ${sliderId}. This assumes Gradio ImageSlider structure.`);
        } else {
            // Fallback to slider-wrap if it exists
            imageWrapper = sliderElement.querySelector('.slider-wrap');
            if (imageWrapper) {
                console.warn(`Using '.slider-wrap' as image wrapper for slider ${sliderId}. Ensure it only contains images.`);
            } else {
                imageWrapper = sliderElement; // Last resort
                console.error(`Could not find a suitable image wrapper (.image-zoom-wrapper, .noUi-base heuristic, .slider-wrap) for slider ${sliderId}. Using slider main element - THIS IS LIKELY INCORRECT.`);
            }
        }
    }

    if (!imageWrapper) {
        console.error(`FATAL: Could not find any suitable image wrapper element for slider ${sliderId}.`);
        cleanupZoomPan(null); // Reset globals
        return;
    }
    console.log(`Found image wrapper for ${sliderId}:`, imageWrapper);


    clipElement = sliderElement.querySelector('.noUi-connect');
    if (!clipElement) {
        console.error(`Could not find clipping element '.noUi-connect' for slider ${sliderId}. Manual clipping will fail.`);
        // Don't return, allow zoom/pan without clipping effect if necessary
    } else {
        clipElement.style.position = 'absolute';
        clipElement.style.top = '0';
        clipElement.style.height = '100%';
        clipElement.style.left = '0';
        console.log(`Found clip element for ${sliderId}:`, clipElement);
    }

    // --- Add listeners to the newly found Image Wrapper ---
    // Use bound functions or ensure 'this' context if needed, but global vars avoid 'this' issues here
    imageWrapper.addEventListener('wheel', handleWheelZoom, { passive: false });
    imageWrapper.addEventListener('mousedown', handlePanStart);

    // Apply necessary styles
    imageWrapper.style.overflow = 'hidden'; // MUST clip the content
    imageWrapper.style.cursor = 'default'; // Initial cursor

    // --- Integrate with noUiSlider ---
    if (sliderElement.noUiSlider) {
        const listenerName = `update.manualClip_${sliderId}`; // Unique listener name
        sliderElement.noUiSlider.off(listenerName); // Remove previous listener for THIS slider
        sliderElement.noUiSlider.on(listenerName, updateClipElementWidth);
        console.log(`Added '${listenerName}' listener to noUiSlider for ${sliderId}.`);
    } else {
        console.warn(`noUiSlider instance not found on sliderElement during init for ${sliderId}.`);
    }

    console.log(`Zoom/Pan listeners added to wrapper for slider ${sliderId}:`, imageWrapper);

    // Reset state and apply initial transform/clipping for THIS slider
    resetZoomPan(true, sliderId); // Explicitly reset the target slider
}


// Function to cleanup listeners and modifications for a SPECIFIC slider
function cleanupZoomPan(sliderIdToClean) {
    const targetSliderId = sliderIdToClean || currentSliderId; // Clean specified or current
    console.log(`Cleaning up zoom and pan for slider: ${targetSliderId}`);

    if (!targetSliderId) {
        console.log("Cleanup: No slider ID specified and no current slider tracked. Resetting globals.");
        // Reset global state variables if no specific slider is targeted
        zoomLevel = 1;
        offsetX = 0;
        offsetY = 0;
        isPanning = false;
        imageWrapper = null;
        sliderElement = null;
        clipElement = null;
        currentSliderId = null;
        // Remove any lingering document listeners
        document.removeEventListener('mousemove', handlePanMove);
        document.removeEventListener('mouseup', handlePanEnd);
        document.removeEventListener('mouseleave', handlePanEnd);
        return;
    }

    const sliderElemToClean = document.getElementById(targetSliderId);
    let wrapperToClean = null;
    let clipElemToClean = null;

    if (sliderElemToClean) {
        // Find the elements again for cleanup, in case globals changed
        wrapperToClean = sliderElemToClean.querySelector('.image-zoom-wrapper')
            || sliderElemToClean.querySelector('.noUi-base') // Check heuristic again
            || sliderElemToClean.querySelector('.slider-wrap')
            || sliderElemToClean; // Fallback
        clipElemToClean = sliderElemToClean.querySelector('.noUi-connect');

        // Remove listeners from the specific wrapper
        if (wrapperToClean) {
            wrapperToClean.removeEventListener('wheel', handleWheelZoom);
            wrapperToClean.removeEventListener('mousedown', handlePanStart);
            wrapperToClean.style.transform = '';
            wrapperToClean.style.cursor = '';
            wrapperToClean.style.userSelect = '';
            wrapperToClean.style.overflow = ''; // Reset overflow
            console.log(`Removed listeners and styles from wrapper of ${targetSliderId}`);
        } else {
            console.warn(`Cleanup: Could not find wrapper for slider ${targetSliderId} to remove listeners.`);
        }

        // Remove specific slider update listener
        if (sliderElemToClean.noUiSlider) {
            const listenerName = `update.manualClip_${targetSliderId}`;
            sliderElemToClean.noUiSlider.off(listenerName);
            console.log(`Removed '${listenerName}' listener from noUiSlider for ${targetSliderId}.`);
        }

        // Reset clip element style override
        if (clipElemToClean) {
            clipElemToClean.style.width = '';
            clipElemToClean.style.position = ''; // May need to reset if originally relative
            // clipElemToClean.style.left = ''; // noUiSlider might reset this anyway
            console.log(`Reset clip element style for ${targetSliderId}`);
        }

    } else {
        console.warn(`Cleanup: Slider element ${targetSliderId} not found.`);
    }

    // Remove document listeners ONLY if the cleaned slider was the one being panned
    // Check isPanning flag and if the cleaned slider matches currentSliderId
    if (isPanning && targetSliderId === currentSliderId) {
        console.log("Cleanup: Removing document listeners as active panning slider is being cleaned.");
        document.removeEventListener('mousemove', handlePanMove);
        document.removeEventListener('mouseup', handlePanEnd);
        document.removeEventListener('mouseleave', handlePanEnd);
        isPanning = false; // Ensure flag is reset
    }

    // If the slider being cleaned up IS the currently active one, reset global state
    if (targetSliderId === currentSliderId) {
        console.log(`Cleanup: Resetting global state as current slider (${currentSliderId}) is being cleaned.`);
        zoomLevel = 1;
        offsetX = 0;
        offsetY = 0;
        isPanning = false; // Ensure reset
        imageWrapper = null;
        sliderElement = null;
        clipElement = null;
        currentSliderId = null; // Mark no slider as active
    }
}


// *** MODIFIED: Generic Fullscreen Toggle Function ***
function toggleSliderFullscreen(sliderId, previewColumnId, fullscreenButtonId, downloadButtonId) {
    console.log(`toggleSliderFullscreen called for slider: ${sliderId}`);
    const previewCol = document.getElementById(previewColumnId);
    if (!previewCol) {
        console.error("Preview column not found:", previewColumnId);
        return;
    }
    const sliderElem = document.getElementById(sliderId);
    if (!sliderElem) {
        console.error("Slider element not found:", sliderId);
        return;
    }

    const isFullscreen = previewCol.classList.contains('full_preview');
    const fullscreenBtn = document.getElementById(fullscreenButtonId);
    const downloadBtn = document.getElementById(downloadButtonId);

    // Always reset zoom/pan state *before* changing visual layout
    // Only reset if *this* slider was the one currently zoomed/panned
    if (currentSliderId === sliderId) {
        resetZoomPan(false, sliderId); // Reset state, don't apply transform yet
    } else {
        console.log(`Slider ${sliderId} wasn't the active zoomed slider (${currentSliderId}), not resetting state.`);
    }


    if (isFullscreen) {
        // Exit fullscreen
        console.log(`Exiting fullscreen for ${sliderId}`);
        previewCol.classList.remove('full_preview');
        if (fullscreenBtn) fullscreenBtn.classList.remove('full');
        if (downloadBtn) downloadBtn.classList.remove('full');
        document.body.style.overflow = 'auto';
        
        // Restore settings panels and other interface elements
        const settingsPanel = document.getElementById('component-37');
        if (settingsPanel) {
            settingsPanel.style.display = '';
        }
        
        // Restore all hidden sibling elements
        const previewParent = previewCol.parentElement;
        if (previewParent) {
            const hiddenElements = previewParent.querySelectorAll('[data-hidden-by-fullscreen="true"]');
            hiddenElements.forEach(element => {
                element.style.display = '';
                element.removeAttribute('data-hidden-by-fullscreen');
            });
        }
        sliderElem.style.zIndex = "";
        // Restore original height - use specific ID if needed
        // TODO: Find a better way to manage original height? Maybe CSS classes?
        if (sliderId === 'compare_slider') {
            sliderElem.style.height = "500px"; // Or read from attribute if set
        } else if (sliderId === 'gallery1') {
            sliderElem.style.height = "400px"; // Or read from attribute
            // Clear any important styles set during fullscreen
            sliderElem.style.removeProperty('width');
            sliderElem.style.removeProperty('max-width');
            sliderElem.style.removeProperty('max-height');
        } else {
            sliderElem.style.height = ""; // Default fallback
        }


        // Reset image rendering only if needed (specific to upscale tab?)
        // Maybe move this logic to Python side if it depends on comparison
        if (sliderId === 'compare_slider') { // Example: only for compare slider
            sliderElem.querySelectorAll('img').forEach(img => {
                img.style.imageRendering = '';
            });
        }

        // Cleanup zoom/pan listeners for *this specific slider*
        console.log(`Cleaning up zoom/pan for ${sliderId} after exiting fullscreen.`);
        cleanupZoomPan(sliderId);

        // Delay layout fixes/re-init (No re-init needed on exit)
        setTimeout(() => {
            window.dispatchEvent(new Event('resize')); // Trigger layout recalc
            if (sliderElem.noUiSlider) {
                // Force slider redraw - crucial after style changes
                sliderElem.noUiSlider.updateOptions({}, false);
                // Re-apply standard clipping logic if needed (noUiSlider should handle this)
                // updateClipElementWidth(); // Might cause issues if noUiSlider resets width %
            }
        }, 50); // Increased delay slightly

    } else {
        // Enter fullscreen
        console.log(`Entering fullscreen for ${sliderId}`);
        previewCol.classList.add('full_preview');
        if (fullscreenBtn) fullscreenBtn.classList.add('full');
        if (downloadBtn) downloadBtn.classList.add('full');
        document.body.style.overflow = 'hidden';
        
        // Hide settings panels and other interface elements
        const settingsPanel = document.getElementById('component-37');
        if (settingsPanel) {
            settingsPanel.style.display = 'none';
        }
        
        // Hide all sibling elements after the preview column
        const previewParent = previewCol.parentElement;
        if (previewParent) {
            let nextSibling = previewCol.nextElementSibling;
            while (nextSibling) {
                nextSibling.style.display = 'none';
                nextSibling.setAttribute('data-hidden-by-fullscreen', 'true');
                nextSibling = nextSibling.nextElementSibling;
            }
        }
        sliderElem.style.zIndex = "1000";
        sliderElem.style.height = "calc(100vh - 80px)"; // Consistent fullscreen height with padding
        sliderElem.style.maxHeight = "calc(100vh - 80px)"; // Ensure max-height is also set
        sliderElem.style.width = "calc(100vw - 80px)"; // Ensure full width
        sliderElem.style.maxWidth = "calc(100vw - 80px)"; // Ensure max-width is also set

        // Apply pixelated rendering if needed (specific to compare tab?)
        if (sliderId === 'compare_slider') { // Example: only for compare slider
            const images = sliderElem.querySelectorAll('img');
            if (images.length === 2) {
                const img1 = images[0];
                const img2 = images[1];
                // Ensure images are loaded before getting natural dimensions
                Promise.all([
                    new Promise(resolve => { if (img1.complete) resolve(); else img1.onload = resolve; }),
                    new Promise(resolve => { if (img2.complete) resolve(); else img2.onload = resolve; })
                ]).then(() => {
                    const img1Area = img1.naturalWidth * img1.naturalHeight;
                    const img2Area = img2.naturalWidth * img2.naturalHeight;
                    if (img1Area > img2Area) {
                        img2.style.imageRendering = 'pixelated'; img1.style.imageRendering = '';
                    } else if (img2Area > img1Area) {
                        img1.style.imageRendering = 'pixelated'; img2.style.imageRendering = '';
                    } else {
                        img1.style.imageRendering = ''; img2.style.imageRendering = '';
                    }
                });
            }
        }

        // Delay layout fixes/re-init
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
            
            // Force correct dimensions regardless of inline styles
            sliderElem.style.setProperty('height', 'calc(100vh - 80px)', 'important');
            sliderElem.style.setProperty('max-height', 'calc(100vh - 80px)', 'important');
            sliderElem.style.setProperty('width', 'calc(100vw - 80px)', 'important');
            sliderElem.style.setProperty('max-width', 'calc(100vw - 80px)', 'important');
            
            if (sliderElem.noUiSlider) {
                sliderElem.noUiSlider.updateOptions({}, false); // Redraw slider first
            }
            // Initialize zoom/pan for *this specific slider* AFTER layout stabilizes
            console.log(`Initializing zoom/pan for ${sliderId} after entering fullscreen.`);
            initializeZoomPan(sliderId);

        }, 50); // Increased delay slightly
    }
}

// *** MODIFIED: ESC Key Handler ***
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' || event.keyCode === 27) {
        console.log("ESC key pressed");

        // Check Compare Tab Fullscreen
        const comparePreviewCol = document.getElementById('compare_preview_column');
        if (comparePreviewCol && comparePreviewCol.classList.contains('full_preview')) {
            console.log("ESC: Exiting Compare fullscreen");
            // Simulate clicking the button to trigger the exit logic and cleanup
            // Find the corresponding button and click it programmatically
            const compareFullscreenBtn = document.getElementById('compare_fullscreen_button');
            if (compareFullscreenBtn) {
                compareFullscreenBtn.click(); // This should trigger toggleSliderFullscreen
            } else {
                // Fallback if button not found (less ideal)
                console.warn("ESC: Compare fullscreen button not found, attempting manual exit.");
                toggleSliderFullscreen('compare_slider', 'compare_preview_column', 'compare_fullscreen_button', 'compare_download_button');
            }
            return; // Exit after handling
        }

        // Check Upscale Tab Fullscreen
        const upscalePreviewCol = document.getElementById('preview_column'); // ID of the upscale preview column
        if (upscalePreviewCol && upscalePreviewCol.classList.contains('full_preview')) {
            console.log("ESC: Exiting Upscale fullscreen");
            // Simulate clicking the button
            const upscaleFullscreenBtn = document.getElementById('fullscreen_button'); // ID of the upscale fullscreen button
            if (upscaleFullscreenBtn) {
                upscaleFullscreenBtn.click(); // This should trigger toggleSliderFullscreen
            } else {
                // Fallback
                console.warn("ESC: Upscale fullscreen button not found, attempting manual exit.");
                toggleSliderFullscreen('gallery1', 'preview_column', 'fullscreen_button', 'download_button');
            }
            return; // Exit after handling
        }
    }
});

// --- IMPORTANT NOTES (REMAIN MOSTLY THE SAME) ---
// 1. HTML Structure: Crucial that both sliders (result_slider/gallery1 and compare_slider)
//    have a compatible inner structure, ideally including a div with class 'image-zoom-wrapper'
//    containing the images, and a '.noUi-connect' element for clipping. The heuristic for '.noUi-base'
//    is a fallback if 'image-zoom-wrapper' isn't present in Gradio's ImageSlider.
//
// 2. Initialization Call: The logic is now primarily handled within toggleSliderFullscreen.
//    No separate initialization call is strictly needed unless you want zoom active outside fullscreen.
//
// 3. Cleanup: Cleanup is handled by toggleSliderFullscreen on exit and by the ESC handler.

// --- END OF FILE compare_fullscreen.js ---