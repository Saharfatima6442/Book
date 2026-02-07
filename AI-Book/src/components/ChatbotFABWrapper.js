import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const ChatbotFABWrapper = () => {
  return (
    <BrowserOnly
      fallback={<div>Loading AI Assistant...</div>}
    >
      {() => {
        const { ChatbotFAB } = require('./Chatbot');
        return <ChatbotFAB backendUrl={process.env.BACKEND_URL || 'http://backend:8000'} />;
      }}
    </BrowserOnly>
  );
};

export default ChatbotFABWrapper;