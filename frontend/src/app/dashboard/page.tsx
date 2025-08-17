'use client';

import React, { useState, useEffect } from 'react';
import { 
  BarChart3, DollarSign, Clock, Zap, TrendingUp, TrendingDown, 
  Settings, Star, Activity, Users, MessageSquare 
} from 'lucide-react';
import { Button } from '@/components/ui/button';

// Mock data - in production this would come from your analytics API
interface AnalyticsData {
  totalRequests: number;
  totalCost: number;
  avgLatency: number;
  costSavings: number;
  modelDistribution: Record<string, number>;
  dailyStats: Array<{
    date: string;
    requests: number;
    cost: number;
    avgLatency: number;
    savings: number;
  }>;
  complexityDistribution: Record<string, number>;
  priorityPreferences: Record<string, number>;
  qualityMetrics: {
    avgQuality: number;
    qualityByModel: Record<string, number>;
  };
}

const mockData: AnalyticsData = {
  totalRequests: 12847,
  totalCost: 45.23,
  avgLatency: 892,
  costSavings: 68.4, // percentage
  modelDistribution: {
    'gpt-3.5-turbo': 45,
    'gpt-4': 15,
    'claude-3-sonnet-20240229': 28,
    'claude-3-haiku-20240307': 12
  },
  dailyStats: [
    { date: '2025-08-01', requests: 1200, cost: 4.2, avgLatency: 850, savings: 65.2 },
    { date: '2025-08-02', requests: 1450, cost: 5.1, avgLatency: 920, savings: 68.1 },
    { date: '2025-08-03', requests: 1680, cost: 6.8, avgLatency: 780, savings: 71.3 },
    { date: '2025-08-04', requests: 1890, cost: 7.2, avgLatency: 845, savings: 69.8 },
    { date: '2025-08-05', requests: 2100, cost: 8.1, avgLatency: 912, savings: 67.5 },
    { date: '2025-08-06', requests: 2240, cost: 8.9, avgLatency: 889, savings: 70.2 },
    { date: '2025-08-07', requests: 2287, cost: 4.9, avgLatency: 876, savings: 72.1 }
  ],
  complexityDistribution: {
    'Low (0-30%)': 45,
    'Medium (30-70%)': 38,
    'High (70-90%)': 14,
    'Very High (90%+)': 3
  },
  priorityPreferences: {
    'speed': 28,
    'cost': 54,
    'quality': 18
  },
  qualityMetrics: {
    avgQuality: 8.4,
    qualityByModel: {
      'gpt-3.5-turbo': 7.8,
      'gpt-4': 9.3,
      'claude-3-sonnet-20240229': 8.7,
      'claude-3-haiku-20240307': 7.9
    }
  }
};

export default function AnalyticsDashboard() {
  const [data, setData] = useState<AnalyticsData>(mockData);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('7d');

  const MetricCard = ({ 
    title, 
    value, 
    change, 
    icon: Icon, 
    color = 'blue',
    subtitle 
  }: {
    title: string;
    value: string;
    change: string;
    icon: React.ElementType;
    color?: string;
    subtitle?: string;
  }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
        </div>
        <div className={`w-12 h-12 bg-${color}-100 rounded-lg flex items-center justify-center`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
      <div className="mt-4 flex items-center">
        {change.startsWith('+') ? (
          <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
        ) : (
          <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
        )}
        <span className={`text-sm ${change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
          {change}
        </span>
        <span className="text-sm text-gray-500 ml-1">vs last period</span>
      </div>
    </div>
  );

  const ModelCard = ({ 
    name, 
    percentage, 
    color, 
    description,
    avgCost,
    avgLatency,
    provider 
  }: {
    name: string;
    percentage: number;
    color: string;
    description: string;
    avgCost: number;
    avgLatency: number;
    provider: string;
  }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="font-medium text-gray-900">{name}</h4>
          <p className="text-xs text-gray-500 capitalize">{provider}</p>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs font-medium bg-${color}-100 text-${color}-800`}>
          {percentage}%
        </span>
      </div>
      
      <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
        <div
          className={`bg-${color}-500 h-2 rounded-full`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      <p className="text-sm text-gray-600 mb-2">{description}</p>
      
      <div className="flex justify-between text-xs text-gray-500">
        <span>Avg Cost: ${avgCost.toFixed(4)}</span>
        <span>Avg Latency: {avgLatency}ms</span>
      </div>
    </div>
  );

  const getModelDisplayName = (modelId: string): string => {
    const modelNames: Record<string, string> = {
      'gpt-3.5-turbo': 'GPT-3.5 Turbo',
      'gpt-4': 'GPT-4',
      'claude-3-sonnet-20240229': 'Claude Sonnet',
      'claude-3-haiku-20240307': 'Claude Haiku'
    };
    return modelNames[modelId] || modelId;
  };

  const getModelProvider = (modelId: string): string => {
    if (modelId.startsWith('gpt-')) return 'OpenAI';
    if (modelId.startsWith('claude-')) return 'Anthropic';
    return 'Unknown';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
              <p className="text-gray-600">Monitor your intelligent LLM routing performance and costs</p>
            </div>
            
            <div className="flex items-center space-x-3">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as '7d' | '30d' | '90d')}
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="90d">Last 90 days</option>
              </select>
              
              <Button variant="outline">
                Export Data
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Total Requests"
            value={data.totalRequests.toLocaleString()}
            change="+12.3%"
            icon={MessageSquare}
            color="blue"
          />
          
          <MetricCard
            title="Total Cost"
            value={`$${data.totalCost.toFixed(2)}`}
            change="-34.2%"
            icon={DollarSign}
            color="green"
            subtitle="68% savings vs GPT-4 only"
          />
          
          <MetricCard
            title="Avg Latency"
            value={`${data.avgLatency}ms`}
            change="-8.1%"
            icon={Clock}
            color="purple"
          />
          
          <MetricCard
            title="Cost Savings"
            value={`${data.costSavings.toFixed(1)}%`}
            change="+2.3%"
            icon={TrendingUp}
            color="orange"
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Daily Requests Chart */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily Requests & Costs</h3>
            <div className="space-y-4">
              {data.dailyStats.slice(-7).map((day, index) => (
                <div key={day.date} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-8 bg-blue-500 rounded-full"></div>
                    <div>
                      <div className="font-medium">{new Date(day.date).toLocaleDateString()}</div>
                      <div className="text-sm text-gray-500">{day.requests.toLocaleString()} requests</div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="font-medium">${day.cost.toFixed(2)}</div>
                    <div className="text-sm text-green-600">{day.savings}% saved</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Complexity Distribution */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Request Complexity Distribution</h3>
            <div className="space-y-4">
              {Object.entries(data.complexityDistribution).map(([complexity, percentage]) => (
                <div key={complexity} className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm font-medium text-gray-700">{complexity}</span>
                    <span className="text-sm text-gray-500">{percentage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Model Distribution */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-6">Model Usage Distribution</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <ModelCard
              name="GPT-3.5 Turbo"
              percentage={data.modelDistribution['gpt-3.5-turbo']}
              color="green"
              description="Fast and cost-effective for general tasks"
              avgCost={0.002}
              avgLatency={500}
              provider="OpenAI"
            />
            
            <ModelCard
              name="Claude Sonnet"
              percentage={data.modelDistribution['claude-3-sonnet-20240229']}
              color="blue"
              description="Balanced performance for analysis and writing"
              avgCost={0.008}
              avgLatency={800}
              provider="Anthropic"
            />
            
            <ModelCard
              name="GPT-4"
              percentage={data.modelDistribution['gpt-4']}
              color="purple"
              description="Maximum capability for complex reasoning"
              avgCost={0.030}
              avgLatency={1200}
              provider="OpenAI"
            />
            
            <ModelCard
              name="Claude Haiku"
              percentage={data.modelDistribution['claude-3-haiku-20240307']}
              color="orange"
              description="Ultra-fast responses for simple queries"
              avgCost={0.005}
              avgLatency={400}
              provider="Anthropic"
            />
          </div>
        </div>

        {/* Priority Preferences & Quality Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* User Priority Preferences */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">User Priority Preferences</h3>
            <div className="space-y-4">
              {Object.entries(data.priorityPreferences).map(([priority, percentage]) => (
                <div key={priority} className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 w-20">
                    {priority === 'speed' && <Zap className="w-4 h-4 text-yellow-500" />}
                    {priority === 'cost' && <DollarSign className="w-4 h-4 text-green-500" />}
                    {priority === 'quality' && <Star className="w-4 h-4 text-blue-500" />}
                    <span className="text-sm font-medium capitalize">{priority}</span>
                  </div>
                  
                  <div className="flex-1">
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          priority === 'speed' ? 'bg-yellow-400' :
                          priority === 'cost' ? 'bg-green-400' : 'bg-blue-400'
                        }`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                  
                  <span className="text-sm text-gray-500 w-10">{percentage}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Quality Metrics */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Metrics</h3>
            
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-lg font-medium">Overall Average Quality</span>
                <span className="text-2xl font-bold text-purple-600">{data.qualityMetrics.avgQuality}/10</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-purple-500 h-3 rounded-full"
                  style={{ width: `${data.qualityMetrics.avgQuality * 10}%` }}
                />
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium text-gray-700">Quality by Model</h4>
              {Object.entries(data.qualityMetrics.qualityByModel).map(([modelId, quality]) => (
                <div key={modelId} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">{getModelDisplayName(modelId)}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-400 h-2 rounded-full"
                        style={{ width: `${quality * 10}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium w-8">{quality}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Cost Optimization Insights */}
        <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl border border-gray-200 p-8">
          <div className="text-center">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">💰 Cost Optimization Insights</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">${(data.totalCost * 3.16).toFixed(2)}</div>
                <div className="text-sm text-gray-600">Would cost with GPT-4 only</div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">${data.totalCost.toFixed(2)}</div>
                <div className="text-sm text-gray-600">Actual cost with routing</div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">${((data.totalCost * 3.16) - data.totalCost).toFixed(2)}</div>
                <div className="text-sm text-gray-600">Total savings</div>
              </div>
            </div>
            
            <p className="text-gray-600 mb-4">
              Your intelligent routing decisions have saved <strong>${((data.totalCost * 3.16) - data.totalCost).toFixed(2)}</strong> 
              ({data.costSavings}%) compared to using GPT-4 for all requests.
            </p>
            
            <div className="flex justify-center">
              <Button>
                <BarChart3 className="w-4 h-4 mr-2" />
                Download Detailed Report
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}