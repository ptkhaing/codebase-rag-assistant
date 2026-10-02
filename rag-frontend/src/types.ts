export interface Source {
  file_path: string;
  start_line: number;
  end_line: number;
}

export interface ChatResponse {
  answer: string;
  rounds_used: number;
  sources: Source[];
}

export type MessageRole = 'user' | 'assistant';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  sources?: Source[];
  roundsUsed?: number;
}