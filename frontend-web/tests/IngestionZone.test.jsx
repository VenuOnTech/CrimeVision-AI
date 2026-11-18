import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => ({ default: { post: vi.fn() } }));

import axios from 'axios';
import IngestionZone from '../src/components/IngestionZone';

const dropFile = (zone, file) =>
  fireEvent.drop(zone, {
    dataTransfer: {
      files: [file],
      items: [
        {
          kind: 'file',
          type: file.type,
          getAsFile: () => file
        }
      ],
      types: ['Files']
    }
  });

describe('IngestionZone', () => {
  beforeEach(() => {
    axios.post.mockReset();
  });

  it('refuses to attach evidence when no case is selected', async () => {
    render(<IngestionZone caseId="" caseName="" />);
    const zone = screen.getByText(/Drop image\/video evidence/i).closest('div');
    const file = new File([new Uint8Array([1, 2, 3])], 'frame.jpg', { type: 'image/jpeg' });
    dropFile(zone, file);
    expect(await screen.findByText(/Select one case before attaching evidence/i)).toBeInTheDocument();
    expect(axios.post).not.toHaveBeenCalled();
  });

  it('disables analysis until a statement is provided', async () => {
    render(<IngestionZone caseId="CASE-A" caseName="A" />);
    const zone = screen.getByText(/Drop image\/video evidence/i).closest('div');
    const file = new File([new Uint8Array([1, 2, 3])], 'frame.jpg', { type: 'image/jpeg' });
    dropFile(zone, file);
    await screen.findByText('frame.jpg');
    expect(screen.getByRole('button', { name: /Run Evidential Analysis/i })).toBeDisabled();
    expect(axios.post).not.toHaveBeenCalled();
  });

  it('uploads and reports INDETERMINATE when the API succeeds', async () => {
    axios.post.mockResolvedValue({
      data: {
        status: 'success',
        case_id: 'CASE-A',
        assessment_status: 'INDETERMINATE',
        graph_status: { backend: 'memory', persisted: false, message: 'volatile' },
      },
    });
    const onUploadSuccess = vi.fn();
    render(<IngestionZone caseId="CASE-A" caseName="A" onUploadSuccess={onUploadSuccess} />);
    const zone = screen.getByText(/Drop image\/video evidence/i).closest('div');
    const file = new File([new Uint8Array([1, 2, 3])], 'frame.jpg', { type: 'image/jpeg' });
    dropFile(zone, file);
    await screen.findByText('frame.jpg');
    fireEvent.change(screen.getByPlaceholderText(/Paste the exact witness statement/i), {
      target: { value: 'A person entered the room.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run Evidential Analysis/i }));
    await waitFor(() => expect(axios.post).toHaveBeenCalledTimes(1));
    expect(
      await screen.findAllByText(/INDETERMINATE/)
    ).toHaveLength(2);
    expect(onUploadSuccess).toHaveBeenCalledTimes(1);
  });
});