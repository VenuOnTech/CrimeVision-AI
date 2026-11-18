import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import GraphView from '../src/components/GraphView';

const statementNode = (status, label) => ({
  id: 'statement_1',
  position: { x: 0, y: 100 },
  data: {
    type: 'statement',
    label,
    full_text: 'hello',
    assessment_status: status,
    case_id: 'CASE-A',
  },
});

describe('GraphView', () => {
  it('updates a statement node when its assessment changes while the id is stable', async () => {
    const { rerender } = render(
      <GraphView nodes={[statementNode('INDETERMINATE', '⚪ "hello"')]} edges={[]} />,
    );
    await waitFor(() => expect(screen.getByText(/⚪ "hello"/)).toBeInTheDocument());

    rerender(<GraphView nodes={[statementNode('CONTRADICTION', '🔴 "hello"')]} edges={[]} />);

    await waitFor(() => expect(screen.getByText(/🔴 "hello"/)).toBeInTheDocument());
    expect(screen.queryByText(/⚪ "hello"/)).not.toBeInTheDocument();
  });
});