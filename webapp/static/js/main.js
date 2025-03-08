// DOM Elements
const userSelect = document.getElementById('userSelect');
const watchedVideosList = document.getElementById('watchedVideosList');
const searchQueriesList = document.getElementById('searchQueriesList');
const getRecommendationsBtn = document.getElementById('getRecommendationsBtn');
const recommendationsGrid = document.getElementById('recommendationsGrid');
const loadingSpinner = document.getElementById('loadingSpinner');
const userPreference = document.getElementById('userPreference');
const preferenceText = document.getElementById('preferenceText');
const videoModal = document.getElementById('videoModal');
const modalVideoTitle = document.getElementById('modalVideoTitle');
const modalVideoDescription = document.getElementById('modalVideoDescription');
const videoWrapper = document.getElementById('videoWrapper');
const videoSource = document.getElementById('videoSource');
const downloadVideoLink = document.getElementById('downloadVideoLink');

// Bootstrap Modal
const videoModalElement = new bootstrap.Modal(videoModal);

// Global Variables
let selectedUser = null;
let userSessions = {};
let watchedVideos = [];
let videoPlayer = null;
let videoSegments = {}; // Store segment information for videos
let thumbnailUrls = {}; // Store thumbnail URLs for videos

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Load user sessions
    loadUserSessions();
    
    // Event listeners
    userSelect.addEventListener('change', handleUserChange);
    getRecommendationsBtn.addEventListener('click', getRecommendations);
});

// Load user sessions from the server
async function loadUserSessions() {
    try {
        const response = await fetch('/get_user_sessions');
        const data = await response.json();
        
        if (data.success) {
            // Store user sessions
            userSessions = data.users.reduce((acc, user) => {
                acc[user.user_id] = user;
                return acc;
            }, {});
            
            // Populate user select dropdown
            populateUserSelect(data.users);
        } else {
            showError('Failed to load user sessions');
        }
    } catch (error) {
        console.error('Error loading user sessions:', error);
        showError('Failed to load user sessions');
    }
}

// Populate user select dropdown
function populateUserSelect(users) {
    // Clear existing options except the first one
    while (userSelect.options.length > 1) {
        userSelect.remove(1);
    }
    
    // Add users to the dropdown
    users.forEach(user => {
        const option = document.createElement('option');
        option.value = user.user_id;
        option.textContent = `User ${user.user_id.substring(0, 8)}...`;
        userSelect.appendChild(option);
    });
}

// Handle user selection change
function handleUserChange() {
    const userId = userSelect.value;
    
    if (userId) {
        selectedUser = userId;
        const user = userSessions[userId];
        
        // Update watched videos list
        updateWatchedVideosList(user);
        
        // Enable get recommendations button
        getRecommendationsBtn.disabled = false;
    } else {
        selectedUser = null;
        resetWatchedVideosList();
        getRecommendationsBtn.disabled = true;
    }
    
    // Reset recommendations
    resetRecommendations();
}

// Update watched videos list
function updateWatchedVideosList(user) {
    // Clear existing list
    watchedVideosList.innerHTML = '';
    
    // Get watched videos
    watchedVideos = [];
    for (let i = 1; i <= 4; i++) {
        const videoKey = `video${i}`;
        if (user[videoKey]) {
            watchedVideos.push(user[videoKey]);
        }
    }
    
    // Add videos to the list
    if (watchedVideos.length > 0) {
        watchedVideos.forEach(video => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = video;
            watchedVideosList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.className = 'list-group-item text-muted';
        li.textContent = 'No videos available';
        watchedVideosList.appendChild(li);
    }
}

// Reset watched videos list
function resetWatchedVideosList() {
    watchedVideosList.innerHTML = '<li class="list-group-item text-muted">No videos selected</li>';
    watchedVideos = [];
}

// Get recommendations from the server
async function getRecommendations() {
    if (!selectedUser || watchedVideos.length === 0) {
        showError('Please select a user with watched videos');
        return;
    }
    
    try {
        // Show loading spinner
        loadingSpinner.classList.remove('d-none');
        recommendationsGrid.classList.add('d-none');
        
        // Make API request
        const response = await fetch('/get_recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: selectedUser,
                session_videos: watchedVideos
            })
        });
        
        const data = await response.json();
        
        // Hide loading spinner
        loadingSpinner.classList.add('d-none');
        recommendationsGrid.classList.remove('d-none');
        
        if (data.success) {
            // Update search queries list
            updateSearchQueriesList(data.user_queries);
            
            // Update user preference
            updateUserPreference(data.preference_summary);
            
            // Store segment information
            videoSegments = data.video_segments || {};
            
            // Store thumbnail URLs
            thumbnailUrls = data.thumbnail_urls || {};
            
            // Display recommendations
            displayRecommendations(data.recommendations, data.video_descriptions);
        } else {
            showError(data.error || 'Failed to get recommendations');
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        loadingSpinner.classList.add('d-none');
        recommendationsGrid.classList.remove('d-none');
        showError('Failed to get recommendations');
    }
}

// Update search queries list
function updateSearchQueriesList(queries) {
    // Clear existing list
    searchQueriesList.innerHTML = '';
    
    // Add queries to the list
    if (queries && queries.length > 0) {
        queries.forEach(query => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = query;
            searchQueriesList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.className = 'list-group-item text-muted';
        li.textContent = 'No queries available';
        searchQueriesList.appendChild(li);
    }
}

// Update user preference
function updateUserPreference(preference) {
    if (preference) {
        preferenceText.textContent = preference;
        userPreference.classList.remove('d-none');
    } else {
        userPreference.classList.add('d-none');
    }
}

// Display recommendations
function displayRecommendations(recommendations, descriptions) {
    // Clear existing recommendations
    recommendationsGrid.innerHTML = '';
    
    console.log("Thumbnail URLs:", thumbnailUrls);
    console.log("Video Segments:", videoSegments);
    
    if (recommendations && recommendations.length > 0) {
        // Create a card for each recommendation
        recommendations.forEach(videoId => {
            const description = descriptions[videoId] || 'No description available';
            
            const col = document.createElement('div');
            col.className = 'col-md-6 col-lg-4 col-xl-3';
            
            const card = document.createElement('div');
            card.className = 'video-card';
            card.dataset.videoId = videoId;
            card.dataset.description = description;
            card.addEventListener('click', () => showVideoDetails(videoId, description));
            
            const thumbnail = document.createElement('div');
            thumbnail.className = 'video-thumbnail';
            
            // Check if we have a thumbnail for this video
            if (thumbnailUrls[videoId]) {
                console.log(`Using thumbnail for ${videoId}: ${thumbnailUrls[videoId]}`);
                // Create an image element for the thumbnail
                const thumbImg = document.createElement('img');
                thumbImg.src = thumbnailUrls[videoId];
                thumbImg.alt = `Thumbnail for ${videoId}`;
                thumbImg.className = 'thumbnail-image';
                thumbnail.appendChild(thumbImg);
                
                // Also set as background for fallback
                thumbnail.style.backgroundImage = `url('${thumbnailUrls[videoId]}')`;
                thumbnail.style.backgroundSize = 'cover';
                thumbnail.style.backgroundPosition = 'center';
            } else {
                console.log(`No thumbnail for ${videoId}, using icon fallback`);
                // Fallback to the icon if no thumbnail is available
                thumbnail.innerHTML = '<i class="fas fa-film"></i>';
            }
            
            // Add segment badge if this video has segment information
            if (videoSegments[videoId]) {
                const segmentBadge = document.createElement('div');
                segmentBadge.className = 'segment-badge';
                segmentBadge.innerHTML = `<i class="fas fa-cut"></i> Segment available`;
                thumbnail.appendChild(segmentBadge);
                
                // Create a video preview element that will be shown on hover
                const previewContainer = document.createElement('div');
                previewContainer.className = 'segment-preview-container';
                
                const previewVideo = document.createElement('video');
                previewVideo.className = 'segment-preview-video';
                previewVideo.muted = true;
                previewVideo.playsInline = true;
                previewVideo.loop = true;
                
                // Set the source to the same video
                const videoUrl = `/video/${videoId}?format=mp4`;
                previewVideo.src = videoUrl;
                
                // Add icon and text overlay
                const previewOverlay = document.createElement('div');
                previewOverlay.className = 'segment-preview-overlay';
                previewOverlay.innerHTML = `
                    <i class="fas fa-play-circle"></i>
                    <span>Showing best segment</span>
                `;
                
                previewContainer.appendChild(previewVideo);
                previewContainer.appendChild(previewOverlay);
                thumbnail.appendChild(previewContainer);
                
                // Add hover event listeners to show/play the preview
                card.addEventListener('mouseenter', () => {
                    // Calculate the correct time to start playing
                    // The user specified that frame numbers actually represent seconds
                    const { start_frame, end_frame } = videoSegments[videoId];
                    const startTime = start_frame; // Frame 1 means 1st second
                    const endTime = end_frame;     // Frame 2 means 2nd second
                    
                    // Load the video and seek to the segment start
                    previewVideo.currentTime = startTime;
                    
                    // Play the segment preview
                    previewVideo.play().then(() => {
                        // Show the preview
                        previewContainer.classList.add('active');
                        
                        // Set up a timeout to stop the preview at the end of the segment
                        const duration = endTime - startTime;
                        if (duration > 0 && duration < 30) { // Limit to 30 seconds max
                            setTimeout(() => {
                                previewVideo.pause();
                            }, duration * 1000);
                        }
                    }).catch(err => {
                        console.error('Error playing preview:', err);
                    });
                });
                
                // Stop and hide preview on mouse leave
                card.addEventListener('mouseleave', () => {
                    previewVideo.pause();
                    previewContainer.classList.remove('active');
                });
            }
            
            const info = document.createElement('div');
            info.className = 'video-info';
            
            const title = document.createElement('h5');
            title.className = 'video-title';
            title.textContent = videoId;
            
            const desc = document.createElement('p');
            desc.className = 'video-description';
            
            // Check if we have segment info to display
            if (videoSegments[videoId]) {
                const { start_frame, end_frame } = videoSegments[videoId];
                // Create a cleaner description without the segment info that we'll display separately
                const cleanDesc = description.replace(`[Best matching segment: seconds ${start_frame}-${end_frame}]`, '');
                desc.textContent = cleanDesc;
                
                // Add segment info as a badge
                const segmentInfo = document.createElement('div');
                segmentInfo.className = 'segment-info mt-2';
                segmentInfo.innerHTML = `<span class="badge bg-primary">Best segment: Seconds ${start_frame}-${end_frame}</span>`;
                info.appendChild(title);
                info.appendChild(desc);
                info.appendChild(segmentInfo);
            } else {
                desc.textContent = description;
                info.appendChild(title);
                info.appendChild(desc);
            }
            
            card.appendChild(thumbnail);
            card.appendChild(info);
            
            col.appendChild(card);
            recommendationsGrid.appendChild(col);
        });
    } else {
        const col = document.createElement('div');
        col.className = 'col-12 text-center text-muted';
        
        const message = document.createElement('p');
        message.textContent = 'No recommendations available';
        
        col.appendChild(message);
        recommendationsGrid.appendChild(col);
    }
}

// Show video details in modal
function showVideoDetails(videoId, description) {
    modalVideoTitle.textContent = videoId;
    modalVideoDescription.textContent = description;
    
    // Handle different video ID formats
    let videoIdForUrl = videoId;
    
    // If the video ID contains a prefix, extract just the number part
    const prefixes = ['scenecliptest', 'sceneclipautoautotrain'];
    for (const prefix of prefixes) {
        if (videoId.includes(prefix)) {
            const match = videoId.match(new RegExp(`${prefix}(\\d+)`));
            if (match && match[1]) {
                // Use the original ID with the prefix to ensure correct format
                videoIdForUrl = videoId;
                break;
            }
        }
    }
    
    // Set video source URLs - Try MP4 first, fallback to original AVI
    const videoUrl = `/video/${videoIdForUrl}?format=mp4`;
    const originalVideoUrl = `/video/${videoIdForUrl}`;  // Original format for download
    console.log(`Loading video from: ${videoUrl}`);
    
    // Set download link to original format
    downloadVideoLink.href = originalVideoUrl;
    
    // Dispose of any existing Video.js player to prevent memory leaks
    if (videoPlayer && typeof videoPlayer.dispose === 'function') {
        try {
            videoPlayer.pause();
            videoPlayer.dispose();
        } catch (err) {
            console.error('Error disposing video player:', err);
        }
        videoPlayer = null;
    }
    
    // Ensure the video player element exists in the DOM
    resetVideoPlayer();
    
    // Set the video source
    const videoSourceEl = document.getElementById('videoSource');
    if (videoSourceEl) {
        videoSourceEl.src = videoUrl;
    }
    
    // Add segment information to the modal if available
    const segmentInfo = videoSegments[videoId];
    if (segmentInfo) {
        // Add segment badge to the modal
        const segmentBadge = document.createElement('div');
        segmentBadge.className = 'segment-control mt-2 mb-3';
        segmentBadge.innerHTML = `
            <button id="jumpToSegmentBtn" class="btn btn-sm btn-primary">
                <i class="fas fa-play-circle"></i> Jump to Best Segment (Seconds ${segmentInfo.start_frame}-${segmentInfo.end_frame})
            </button>
        `;
        // Insert after the video-container
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.after(segmentBadge);
        }
    }
    
    // Show the modal first
    videoModalElement.show();
    
    // Initialize VideoJS only after the modal is fully shown
    videoModal.addEventListener('shown.bs.modal', function initializePlayer() {
        // Remove this event listener to avoid duplicate initializations
        videoModal.removeEventListener('shown.bs.modal', initializePlayer);
        
        // Ensure the element exists before initializing
        const playerElement = document.getElementById('videoPlayer');
        if (!playerElement) {
            console.error('Video player element not found in the DOM');
            resetVideoPlayer(); // Try to reset the player again
            return;
        }
        
        // Initialize Video.js with codec options
        try {
            videoPlayer = videojs('videoPlayer', {
                controls: true,
                autoplay: false,
                preload: 'auto',
                fluid: true,
                playbackRates: [0.5, 1, 1.5, 2],
                sources: [{
                    src: videoUrl,
                    type: 'video/mp4'
                }],
                techOrder: ["html5"]
            });
            
            // Setup jump to segment button if segment info is available
            if (segmentInfo) {
                const jumpBtn = document.getElementById('jumpToSegmentBtn');
                if (jumpBtn) {
                    jumpBtn.addEventListener('click', function() {
                        // According to the user, frame numbers represent seconds directly
                        const startTimeSeconds = segmentInfo.start_frame;
                        
                        // Jump to the segment start time
                        if (videoPlayer && videoPlayer.currentTime) {
                            videoPlayer.currentTime(startTimeSeconds);
                            videoPlayer.play();
                        }
                    });
                }
            }
            
            // Add error handling
            videoPlayer.on('error', function() {
                console.error('Video.js error:', videoPlayer.error());
                
                // Show error message with download option
                const errorDisplay = document.createElement('div');
                errorDisplay.className = 'video-error-message';
                errorDisplay.innerHTML = `
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Sorry, this video cannot be played in your browser.</p>
                    <p>Please use the download button below to view the video on your device.</p>
                `;
                
                // Replace the video player with error message
                videoWrapper.innerHTML = '';
                videoWrapper.appendChild(errorDisplay);
            });
        } catch (err) {
            console.error('Error initializing video player:', err);
            
            // Show error message with download option
            const errorDisplay = document.createElement('div');
            errorDisplay.className = 'video-error-message';
            errorDisplay.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <p>Sorry, there was an error initializing the video player.</p>
                <p>Please use the download button below to view the video on your device.</p>
            `;
            
            // Replace the video player with error message
            videoWrapper.innerHTML = '';
            videoWrapper.appendChild(errorDisplay);
        }
    }, { once: true });
    
    // Add event listener for when modal is closed
    videoModal.addEventListener('hidden.bs.modal', function () {
        // Dispose of the Video.js player to prevent memory leaks
        if (videoPlayer && typeof videoPlayer.dispose === 'function') {
            try {
                videoPlayer.pause();
                videoPlayer.dispose();
            } catch (err) {
                console.error('Error disposing video player:', err);
            }
            videoPlayer = null;
        }
        
        // Remove segment badge if it exists
        const segmentBadge = document.querySelector('.segment-control');
        if (segmentBadge) {
            segmentBadge.remove();
        }
        
        // Reset the video player element to ensure it exists for next time
        resetVideoPlayer();
    }, { once: true });
}

// Helper function to reset the video player element
function resetVideoPlayer() {
    // Recreate the video element
    videoWrapper.innerHTML = `
        <video id="videoPlayer" class="video-js vjs-default-skin vjs-big-play-centered" controls preload="auto" width="640" height="400">
            <source id="videoSource" src="" type="video/mp4">
            <p class="vjs-no-js">
                To view this video please enable JavaScript, and consider upgrading to a
                web browser that <a href="https://videojs.com/html5-video-support/" target="_blank">supports HTML5 video</a>
            </p>
        </video>
    `;
}

// Reset recommendations
function resetRecommendations() {
    recommendationsGrid.innerHTML = '<div class="col-12 text-center text-muted"><p>Select a user and click "Get Recommendations" to see personalized video suggestions</p></div>';
    userPreference.classList.add('d-none');
    searchQueriesList.innerHTML = '<li class="list-group-item text-muted">No queries available</li>';
}

// Show error message
function showError(message) {
    alert(message);
} 