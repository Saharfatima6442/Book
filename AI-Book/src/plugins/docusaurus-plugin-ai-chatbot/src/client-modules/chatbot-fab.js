import { ChatbotFAB } from '@site/src/components/Chatbot';
import React from 'react';
import { createRoot } from 'react-dom/client';

// Create a container element for the chatbot
const createChatbotContainer = () => {
  const container = document.createElement('div');
  container.id = 'ai-chatbot-container';
  container.style.zIndex = '1000';
  document.body.appendChild(container);
  return container;
};

// Initialize the chatbot when the DOM is ready
const initializeChatbot = () => {
  const container = createChatbotContainer();
  const root = createRoot(container);

  // Use the backend URL from environment or default to localhost
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';

  root.render(<ChatbotFAB backendUrl={backendUrl} />);
};

// Wait for the DOM to be fully loaded before initializing
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeChatbot);
  } else {
    // DOM is already ready
    initializeChatbot();
  }
}

// Export nothing since this is just for side effects
export default undefined;