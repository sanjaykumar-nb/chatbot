document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');

    // Helper function to add a message to the chat window
    const addMessage = (sender, text) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        messageElement.innerText = text;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
    };

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        // Display the user's message
        addMessage('user', userMessage);
        messageInput.value = '';

        // Display a loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('message', 'bot-message', 'loading-message');
        loadingElement.innerText = 'Thinking...';
        chatMessages.appendChild(loadingElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // Construct the API URL - works for both local and deployed environments
            const apiUrl = `${window.location.origin}/chat`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: userMessage }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Remove the loading indicator
            chatMessages.removeChild(loadingElement);

            // Display the bot's response
            addMessage('bot', data.answer);

        } catch (error) {
            // Remove loading indicator and show an error
             chatMessages.removeChild(loadingElement);
            addMessage('bot', 'Sorry, something went wrong. Please try again.');
            console.error('Error fetching chat response:', error);
        }
    });

    // Add initial welcome message
    addMessage('bot', 'Hello! How can I assist you today?');
});