import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import Heatmap from '../src/components/Heatmap';

describe('Heatmap', () => {
  it('renders empty state when no matrix is available', () => {
    render(<Heatmap />);
    expect(screen.getByText(/No stored patch\/token matrix/i)).toBeInTheDocument();
  });

  it('renders tokens, patches, and classifier assessment when matrix present', () => {
    render(
      <Heatmap
        data={[[0.1, 0.2], [0.3, 0.4]]}
        tokens={['alpha', 'beta']}
        patches={['p1', 'p2']}
        assessmentStatus="MATCH"
        evidenceName="test.jpg"
        cost={0.5}
      />,
    );
    expect(screen.getByText('alpha')).toBeInTheDocument();
    expect(screen.getByText('beta')).toBeInTheDocument();
    expect(screen.getAllByText('MATCH').length).toBeGreaterThan(0);
  });

  it('renders INDETERMINATE for unknown status', () => {
    render(
      <Heatmap
        data={[[0.1]]}
        tokens={['x']}
        patches={['p']}
        assessmentStatus="MAYBE"
      />,
    );
    expect(screen.getAllByText('INDETERMINATE').length).toBeGreaterThan(0);
  });
});