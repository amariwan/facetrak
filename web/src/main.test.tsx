import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { App } from './app';

describe('dashboard', () => {
  it('renders all sections and dark mode toggle', () => {
    render(<App />);
    expect(screen.getByText('FaceTrak 1.0')).toBeTruthy();
    for (const s of ['Cameras', 'Hardware', 'Diagnostics'])
      expect(screen.getByText(s)).toBeTruthy();
    expect(screen.getByLabelText('toggle dark mode')).toBeTruthy();
  });
});
