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

// Bootstrap Modal
const videoModalElement = new bootstrap.Modal(videoModal);

// Global Variables
let selectedUser = null;
let userSessions = {};
let watchedVideos = [];

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
            thumbnail.innerHTML = '<i class="fas fa-film"></i>';
            
            const info = document.createElement('div');
            info.className = 'video-info';
            
            const title = document.createElement('h5');
            title.className = 'video-title';
            title.textContent = videoId;
            
            const desc = document.createElement('p');
            desc.className = 'video-description';
            desc.textContent = description;
            
            info.appendChild(title);
            info.appendChild(desc);
            
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
    videoModalElement.show();
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