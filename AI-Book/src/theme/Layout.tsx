import React from 'react';
import Chatbot from '@site/src/components/Chatbot';
import OriginalLayout from '@theme-original/Layout';

// Use your actual Hugging Face Space URL when deploying
const CHATBOT_BACKEND_URL = process.env.REACT_APP_CHATBOT_URL || 'https://your-hf-space-url.hf.space';

type LayoutProps = {
  children: React.ReactNode;
  [key: string]: any;
};

const LayoutWrapper = (props: LayoutProps): JSX.Element => {
  return (
    <>
      <OriginalLayout {...props} />
      <Chatbot backendUrl={CHATBOT_BACKEND_URL} />
    </>
  );
};

export default LayoutWrapper;