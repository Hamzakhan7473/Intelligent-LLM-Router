import React from 'react';
import LLMRouterInterface from '@/components/chat/ChatInterface';

export const metadata = {
  title: 'LLM Router - Test Intelligent Model Selection',
  description: 'Experience intelligent routing between GPT-4, GPT-3.5, and Claude models based on task type, complexity, and performance preferences',
};

export default function LLMRouterPage() {
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <LLMRouterInterface />
    </div>
  );
}