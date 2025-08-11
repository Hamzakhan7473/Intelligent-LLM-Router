import React from 'react';
import ChatInterface from '@/components/chat/ChatInterface';

export const metadata = {
  title: 'Chat - Intelligent LLM Router',
  description: 'Test the intelligent LLM routing system with real-time chat interface',
};

export default function ChatPage() {
  return (
    <div className="h-screen">
      <ChatInterface />
    </div>
  );
}