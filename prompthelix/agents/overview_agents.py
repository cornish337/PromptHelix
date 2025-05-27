# Advanced AI Agents Suite for PromptHelix System
# ==================================================

# File: agents/base.py
"""
Base Agent Architecture
Provides the foundational structure for all specialized agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time
import uuid
from datetime import datetime

class AgentType(Enum):
    CRITIC = "critic"
    DOMAIN_EXPERT = "domain_expert"
    META_LEARNER = "meta_learner"
    RESULTS_EVALUATOR = "results_evaluator"
    STYLE_OPTIMIZER = "style_optimizer"

@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 5=high

@dataclass
class AgentResponse:
    """Standardized response format from agents."""
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the system.
    
    Provides common functionality including:
    - Message handling and communication protocols
    - Performance monitoring and logging
    - Configuration management
    - Health checking and status reporting
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_type = self._get_agent_type()
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
        self.message_queue = asyncio.Queue()
        self.performance_metrics = {
            'messages_processed': 0,
            'average_response_time': 0.0,
            'success_rate': 0.0,
            'last_activity': None
        }
        self.is_active = False
        self._setup_agent()
    
    @abstractmethod
    def _get_agent_type(self) -> AgentType:
        """Return the specific agent type."""
        pass
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message and return a response."""
        pass
    
    def _setup_agent(self):
        """Initialize agent-specific components."""
        self.logger.info(f"Initializing {self.agent_type.value} agent: {self.agent_id}")
    
    async def start(self):
        """Start the agent's main processing loop."""
        self.is_active = True
        self.logger.info(f"Agent {self.agent_id} started")
        
        while self.is_active:
            try:
                # Process messages from queue
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
    
    async def stop(self):
        """Stop the agent gracefully."""
        self.is_active = False
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send a message to this agent."""
        await self.message_queue.put(message)
        return AgentResponse(success=True, data={"status": "message_queued"})
    
    async def _handle_message(self, message: AgentMessage) -> AgentResponse:
        """Internal message handling with performance tracking."""
        start_time = time.time()
        
        try:
            response = await self.process_message(message)
            response.processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_metrics(response.success, response.processing_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return AgentResponse(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update agent performance metrics."""
        self.performance_metrics['messages_processed'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        count = self.performance_metrics['messages_processed']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (count - 1) + processing_time) / count
        )
        
        # Update success rate
        if success:
            current_success_rate = self.performance_metrics['success_rate']
            self.performance_metrics['success_rate'] = (
                (current_success_rate * (count - 1) + 1.0) / count
            )
        
        self.performance_metrics['last_activity'] = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and performance metrics."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'is_active': self.is_active,
            'performance_metrics': self.performance_metrics,
            'queue_size': self.message_queue.qsize()
        }

# ==================================================
# File: agents/critic.py
"""
Critic Agent - Analyzes and critiques prompts using reinforcement learning principles.
Implements Actor-Critic architecture for prompt evaluation and improvement suggestions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from .base import BaseAgent, AgentType, AgentMessage, AgentResponse

class CriticNetwork(nn.Module):
    """Neural network for prompt criticism and evaluation."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single value output for criticism score
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class CriticAgent(BaseAgent):
    """
    Advanced prompt critic using Actor-Critic methodology.
    
    Responsibilities:
    - Analyze prompt quality across multiple dimensions
    - Provide constructive criticism and improvement suggestions
    - Learn from feedback to improve criticism accuracy
    - Maintain criticism history for pattern recognition
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.embedding_dim = config.get('embedding_dim', 512) if config else 512
        self.learning_rate = config.get('learning_rate', 0.001) if config else 0.001
        
        super().__init__(agent_id, config)
        
        # Initialize neural networks
        self.critic_net = CriticNetwork(self.embedding_dim)
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Criticism categories and weights
        self.criticism_categories = {
            'clarity': 0.25,
            'specificity': 0.20,
            'completeness': 0.20,
            'effectiveness': 0.15,
            'creativity': 0.10,
            'bias_potential': 0.10
        }
        
        # Historical data for learning
        self.criticism_history = []
        self.feedback_buffer = []
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.CRITIC
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming criticism requests."""
        message_type = message.message_type
        
        if message_type == "critique_prompt":
            return await self._critique_prompt(message.payload)
        elif message_type == "update_feedback":
            return await self._update_from_feedback(message.payload)
        elif message_type == "get_criticism_patterns":
            return await self._get_criticism_patterns()
        else:
            return AgentResponse(
                success=False,
                error_message=f"Unknown message type: {message_type}"
            )
    
    async def _critique_prompt(self, payload: Dict[str, Any]) -> AgentResponse:
        """Provide comprehensive criticism of a prompt."""
        try:
            prompt_text = payload.get('prompt')
            context = payload.get('context', {})
            
            if not prompt_text:
                return AgentResponse(
                    success=False,
                    error_message="No prompt provided for criticism"
                )
            
            # Generate prompt embedding (simplified - in practice, use proper embeddings)
            prompt_embedding = self._generate_prompt_embedding(prompt_text)
            
            # Get neural network criticism score
            with torch.no_grad():
                embedding_tensor = torch.FloatTensor(prompt_embedding).unsqueeze(0)
                neural_score = self.critic_net(embedding_tensor).item()
            
            # Perform multi-dimensional analysis
            detailed_criticism = self._analyze_prompt_dimensions(prompt_text, context)
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                prompt_text, detailed_criticism
            )
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(detailed_criticism, neural_score)
            
            criticism_result = {
                'overall_score': overall_score,
                'neural_score': neural_score,
                'detailed_analysis': detailed_criticism,
                'improvement_suggestions': suggestions,
                'criticism_confidence': min(0.9, overall_score + 0.1)
            }
            
            # Store for learning
            self.criticism_history.append({
                'prompt': prompt_text,
                'criticism': criticism_result,
                'timestamp': message.timestamp
            })
            
            return AgentResponse(
                success=True,
                data=criticism_result,
                confidence=criticism_result['criticism_confidence']
            )
            
        except Exception as e:
            self.logger.error(f"Error in prompt criticism: {e}")
            return AgentResponse(
                success=False,
                error_message=f"Criticism failed: {str(e)}"
            )
    
    def _generate_prompt_embedding(self, prompt: str) -> List[float]:
        """Generate embedding representation of prompt (simplified implementation)."""
        # In production, use proper embedding models (e.g., sentence-transformers)
        words = prompt.lower().split()
        
        # Simple bag-of-words with semantic features
        features = {
            'length': len(prompt),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'question_words': sum(1 for w in words if w in ['what', 'how', 'why', 'when', 'where']),
            'imperative_words': sum(1 for w in words if w in ['create', 'generate', 'write', 'explain']),
            'complexity_markers': sum(1 for w in words if w in ['complex', 'detailed', 'comprehensive'])
        }
        
        # Pad to embedding dimension
        embedding = [features.get(f'feature_{i}', 0) for i in range(self.embedding_dim)]
        
        # Normalize
        embedding = np.array(embedding[:self.embedding_dim])
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def _analyze_prompt_dimensions(self, prompt: str, context: Dict) -> Dict[str, Dict]:
        """Analyze prompt across multiple quality dimensions."""
        analysis = {}
        
        # Clarity analysis
        analysis['clarity'] = self._analyze_clarity(prompt)
        
        # Specificity analysis
        analysis['specificity'] = self._analyze_specificity(prompt)
        
        # Completeness analysis
        analysis['completeness'] = self._analyze_completeness(prompt, context)
        
        # Effectiveness analysis
        analysis['effectiveness'] = self._analyze_effectiveness(prompt)
        
        # Creativity analysis
        analysis['creativity'] = self._analyze_creativity(prompt)
        
        # Bias potential analysis
        analysis['bias_potential'] = self._analyze_bias_potential(prompt)
        
        return analysis
    
    def _analyze_clarity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt clarity."""
        words = prompt.split()
        sentences = prompt.split('.')
        
        # Simple clarity metrics
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        complex_words = sum(1 for w in words if len(w) > 8)
        
        clarity_score = max(0, min(1, 1 - (avg_sentence_length / 20) - (complex_words / len(words))))
        
        return {
            'score': clarity_score,
            'issues': [] if clarity_score > 0.7 else ['Long sentences detected', 'Complex vocabulary'],
            'suggestions': [] if clarity_score > 0.7 else ['Use shorter sentences', 'Simplify vocabulary']
        }
    
    def _analyze_specificity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt specificity."""
        vague_words = ['something', 'anything', 'stuff', 'things', 'good', 'nice', 'some']
        specific_words = ['exactly', 'specifically', 'precisely', 'detailed', 'comprehensive']
        
        vague_count = sum(1 for word in prompt.lower().split() if word in vague_words)
        specific_count = sum(1 for word in prompt.lower().split() if word in specific_words)
        
        specificity_score = max(0, min(1, (specific_count - vague_count) / max(1, len(prompt.split())) + 0.5))
        
        return {
            'score': specificity_score,
            'vague_words_found': vague_count,
            'specific_indicators': specific_count,
            'suggestions': ['Add more specific requirements', 'Define vague terms'] if specificity_score < 0.6 else []
        }
    
    def _analyze_completeness(self, prompt: str, context: Dict) -> Dict[str, Any]:
        """Analyze prompt completeness."""
        required_elements = ['task', 'format', 'constraints']
        found_elements = []
        
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['write', 'create', 'generate', 'explain']):
            found_elements.append('task')
        if any(word in prompt_lower for word in ['format', 'style', 'structure']):
            found_elements.append('format')
        if any(word in prompt_lower for word in ['not', 'avoid', 'must', 'should', 'limit']):
            found_elements.append('constraints')
        
        completeness_score = len(found_elements) / len(required_elements)
        
        return {
            'score': completeness_score,
            'found_elements': found_elements,
            'missing_elements': [elem for elem in required_elements if elem not in found_elements],
            'suggestions': [f'Add {elem} specification' for elem in required_elements if elem not in found_elements]
        }
    
    def _analyze_effectiveness(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt effectiveness potential."""
        action_words = ['create', 'write', 'explain', 'analyze', 'compare', 'evaluate']
        context_words = ['because', 'for', 'since', 'given', 'assuming']
        
        action_count = sum(1 for word in prompt.lower().split() if word in action_words)
        context_count = sum(1 for word in prompt.lower().split() if word in context_words)
        
        effectiveness_score = min(1, (action_count * 0.3 + context_count * 0.2 + 0.5))
        
        return {
            'score': effectiveness_score,
            'action_indicators': action_count,
            'context_indicators': context_count,
            'suggestions': ['Add clear action words', 'Provide more context'] if effectiveness_score < 0.7 else []
        }
    
    def _analyze_creativity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt creativity potential."""
        creative_words = ['creative', 'innovative', 'unique', 'original', 'imaginative', 'novel']
        constraint_words = ['exactly', 'precisely', 'must', 'only', 'strictly']
        
        creative_count = sum(1 for word in prompt.lower().split() if word in creative_words)
        constraint_count = sum(1 for word in prompt.lower().split() if word in constraint_words)
        
        creativity_score = max(0, min(1, (creative_count * 0.4 - constraint_count * 0.2 + 0.5)))
        
        return {
            'score': creativity_score,
            'creative_indicators': creative_count,
            'constraint_level': constraint_count,
            'suggestions': ['Encourage creative thinking', 'Allow for multiple approaches'] if creativity_score < 0.5 else []
        }
    
    def _analyze_bias_potential(self, prompt: str) -> Dict[str, Any]:
        """Analyze potential bias in prompt."""
        bias_indicators = ['he', 'she', 'guys', 'girls', 'normal', 'typical', 'obviously']
        inclusive_words = ['they', 'people', 'individuals', 'persons', 'everyone']
        
        bias_count = sum(1 for word in prompt.lower().split() if word in bias_indicators)
        inclusive_count = sum(1 for word in prompt.lower().split() if word in inclusive_words)
        
        bias_score = max(0, min(1, 1 - (bias_count * 0.3) + (inclusive_count * 0.2)))
        
        return {
            'score': bias_score,
            'bias_indicators': bias_count,
            'inclusive_language': inclusive_count,
            'suggestions': ['Use more inclusive language', 'Avoid assumptions'] if bias_score < 0.7 else []
        }
    
    def _calculate_overall_score(self, analysis: Dict, neural_score: float) -> float:
        """Calculate weighted overall quality score."""
        category_scores = {category: analysis[category]['score'] for category in self.criticism_categories}
        
        weighted_score = sum(
            score * self.criticism_categories[category]
            for category, score in category_scores.items()
        )
        
        # Combine with neural network score
        final_score = (weighted_score * 0.7) + (neural_score * 0.3)
        
        return round(final_score, 3)
    
    def _generate_improvement_suggestions(self, prompt: str, analysis: Dict) -> List[str]:
        """Generate specific improvement suggestions based on analysis."""
        suggestions = []
        
        for category, details in analysis.items():
            if details['score'] < 0.7:  # Threshold for improvement
                suggestions.extend(details.get('suggestions', []))
        
        # Add general suggestions based on overall patterns
        if len(prompt.split()) < 10:
            suggestions.append("Consider adding more detail to your prompt")
        
        if '?' not in prompt and not any(word in prompt.lower() for word in ['create', 'write', 'generate']):
            suggestions.append("Make your request more explicit")
        
        return list(set(suggestions))  # Remove duplicates
    
    async def _update_from_feedback(self, payload: Dict[str, Any]) -> AgentResponse:
        """Update critic model based on feedback."""
        try:
            feedback_data = payload.get('feedback', {})
            expected_score = feedback_data.get('expected_score')
            prompt_embedding = feedback_data.get('prompt_embedding')
            
            if expected_score is not None and prompt_embedding is not None:
                # Convert to tensors
                embedding_tensor = torch.FloatTensor(prompt_embedding).unsqueeze(0)
                target_tensor = torch.FloatTensor([expected_score])
                
                # Forward pass
                predicted_score = self.critic_net(embedding_tensor)
                
                # Calculate loss and backpropagate
                loss = self.criterion(predicted_score, target_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                return AgentResponse(
                    success=True,
                    data={'loss': loss.item(), 'updated': True}
                )
            
            return AgentResponse(
                success=False,
                error_message="Invalid feedback data provided"
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error_message=f"Feedback update failed: {str(e)}"
            )
    
    async def _get_criticism_patterns(self) -> AgentResponse:
        """Return patterns learned from criticism history."""
        try:
            if not self.criticism_history:
                return AgentResponse(
                    success=True,
                    data={'patterns': [], 'message': 'No criticism history available'}
                )
            
            # Analyze patterns in criticism history
            patterns = {
                'common_issues': self._find_common_issues(),
                'improvement_trends': self._analyze_improvement_trends(),
                'category_statistics': self._calculate_category_statistics()
            }
            
            return AgentResponse(
                success=True,
                data={'patterns': patterns}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error_message=f"Pattern analysis failed: {str(e)}"
            )
    
    def _find_common_issues(self) -> List[Dict[str, Any]]:
        """Find most common issues in criticized prompts."""
        issue_counts = {}
        
        for record in self.criticism_history:
            analysis = record['criticism']['detailed_analysis']
            for category, details in analysis.items():
                for suggestion in details.get('suggestions', []):
                    issue_counts[suggestion] = issue_counts.get(suggestion, 0) + 1
        
        return [
            {'issue': issue, 'frequency': count}
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze trends in prompt quality over time."""
        if len(self.criticism_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_scores = [
            record['criticism']['overall_score']
            for record in self.criticism_history[-10:]
        ]
        
        if len(recent_scores) > 1:
            trend = 'improving' if recent_scores[-1] > recent_scores[0] else 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'average_recent_score': np.mean(recent_scores),
            'score_variance': np.var(recent_scores)
        }
    
    def _calculate_category_statistics(self) -> Dict[str, float]:
        """Calculate average scores by category."""
        category_stats = {}
        
        for category in self.criticism_categories:
            scores = [
                record['criticism']['detailed_analysis'][category]['score']
                for record in self.criticism_history
                if category in record['criticism']['detailed_analysis']
            ]
            
            if scores:
                category_stats[category] = {
                    'average_score': np.mean(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'std_dev': np.std(scores)
                }
        
        return category_stats

# ==================================================
# File: agents/domain_expert.py
"""
Domain Expert Agent - Provides specialized knowledge and domain-specific optimization.
Implements adaptive expertise across multiple domains with knowledge retrieval and reasoning.
"""

import json
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from .base import BaseAgent, AgentType, AgentMessage, AgentResponse

class KnowledgeBase:
    """Structured knowledge base for domain expertise."""
    
    def __init__(self):
        self.domains = {}
        self.domain_relationships = defaultdict(set)
        self.expertise_levels = {}
        self.knowledge_graph = defaultdict(dict)
    
    def add_domain(self, domain: str, knowledge: Dict[str, Any], expertise_level: float = 0.5):
        """Add or update domain knowledge."""
        self.domains[domain] = knowledge
        self.expertise_levels[domain] = expertise_level
    
    def get_domain_knowledge(self, domain: str) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge for a specific domain."""
        return self.domains.get(domain)
    
    def find_related_domains(self, domain: str) -> Set[str]:
        """Find domains related to the given domain."""
        return self.domain_relationships.get(domain, set())
    
    def add_relationship(self, domain1: str, domain2: str, strength: float = 0.5):
        """Add relationship between domains."""
        self.domain_relationships[domain1].add(domain2)
        self.domain_relationships[domain2].add(domain1)
        self.knowledge_graph[domain1][domain2] = strength
        self.knowledge_graph[domain2][domain1] = strength

class DomainExpertAgent(BaseAgent):
    """
    Advanced domain expert providing specialized knowledge and optimization.
    
    Responsibilities:
    - Maintain expertise across multiple domains
    - Provide domain-specific prompt optimization
    - Adapt knowledge based on context and feedback
    - Cross-domain knowledge transfer and synthesis
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        self.knowledge_base = KnowledgeBase()
        self.active_domains = set()
        self.domain_usage_stats = defaultdict(int)
        self.expertise_threshold = config.get('expertise_threshold', 0.7) if config else 0.7
        
        # Initialize with common domains
        self._initialize_default_domains()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.knowledge_decay_rate = 0.05  # How fast unused knowledge decays
        self.adaptation_history = []
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.DOMAIN_EXPERT
    
    def _initialize_default_domains(self):
        """Initialize knowledge base with common domains."""
        domains_data = {
            'technology': {
                'keywords': ['AI', 'machine learning', 'software', 'programming', 'algorithm', 'data'],
                'best_practices': [
                    'Use specific technical terminology',
                    'Include version numbers and specifications',
                    'Consider scalability and performance',
                    'Address security implications'
                ],
                'common_patterns': [
                    'Step-by-step implementation guides',
                    'Code examples with explanations',
                    'Architecture diagrams and documentation',
                    'Error handling and edge cases'
                ],
                'prompt_templates': {
                    'code_generation': 'Generate {language} code that {task}. Include error handling and comments.',
                    'architecture_design': 'Design a {system_type} architecture for {requirements}. Consider scalability and maintainability.',
                    'debugging': 'Debug this {language} code: {code}. Identify issues and provide fixes.'
                }
            },
            'business': {
                'keywords': ['strategy', 'market', 'revenue', 'growth', 'competition', 'ROI'],
                'best_practices': [
                    'Focus on measurable outcomes',
                    'Consider market context and competition',
                    'Include risk assessment',
                    'Align with business objectives'
                ],
                'common_patterns': [
                    'SWOT analysis frameworks',
                    'Financial projections and models',
                    'Market research and analysis',
                    'Strategic planning processes'
                ],
                'prompt_templates': {
                    'market_analysis': 'Analyze the market for {product/service} including size, trends, and competition.',
                    'business_plan': 'Create a business plan for {business_idea} including financial projections and go-to-market strategy.',
                    'strategy_development': 'Develop a strategic plan for {objective} considering current market conditions.'
                }
            },
            'creative': {
                'keywords': ['story', 'creative', 'artistic', 'design', 'narrative', 'imagination'],
                'best_practices': [
                    'Encourage originality and uniqueness',
                    'Allow for multiple interpretations',
                    'Focus on emotional engagement',
                    'Consider audience and context'
                ],
                'common_patterns': [
                    'Character development and backstory',
                    'Plot structure and narrative arc',
                    'Visual composition and aesthetics',
                    'Thematic elements and symbolism'
                ],
                'prompt_templates': {
                    'story_writing': 'Write a {genre} story about {premise}. Focus on character development and {theme}.',
                    'creative_design': 'Design a {type} that {purpose}. Consider aesthetics, functionality, and user experience.',
                    'content_creation': 'Create {content_type} for {audience} that {objective}.'
                }
            },
            'education': {
                'keywords': ['learning', 'teaching', 'curriculum', 'assessment', 'pedagogy', 'knowledge'],
                'best_practices': [
                    'Consider learning objectives and outcomes',
                    'Adapt to different learning styles',
                    'Include assessment and feedback mechanisms',
                    'Ensure content accessibility'
                ],
                'common_patterns': [
                    'Lesson planning and structure',
                    'Interactive learning activities',
                    'Assessment rubrics and criteria',
                    'Differentiated instruction approaches'
                ],
                'prompt_
