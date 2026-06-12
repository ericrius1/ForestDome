export interface CursorTarget {
  color: string;
}

export interface Cursor {
  /** Built-in targets; add more as new interactables appear. */
  targets: Record<string, CursorTarget>;
  /** Point at a registered target (or null for neutral). `active` = engaged (e.g. mouse held). */
  set(target: string | null, active?: boolean): void;
  dispose(): void;
}

const NEUTRAL = 'rgba(255,255,255,0.5)';

export function createCursor(): Cursor {
  const targets: Record<string, CursorTarget> = {
    attract: { color: '#4fd8ff' },
    repulse: { color: '#ffa14f' },
  };

  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed; left:50%; top:50%; width:18px; height:18px;
    transform:translate(-50%,-50%); border:2px solid ${NEUTRAL};
    border-radius:50%; box-sizing:border-box; pointer-events:none; z-index:30;
    transition:border-color .12s, box-shadow .12s, transform .1s;`;
  const dot = document.createElement('div');
  dot.style.cssText = `
    position:absolute; left:50%; top:50%; width:3px; height:3px; margin:-1.5px;
    border-radius:50%; background:rgba(255,255,255,0.9);`;
  el.appendChild(dot);
  document.body.appendChild(el);

  function set(target: string | null, active = false) {
    const t = target ? targets[target] : undefined;
    el.style.borderColor = t ? t.color : NEUTRAL;
    el.style.boxShadow = t ? `0 0 8px ${t.color}` : 'none';
    el.style.transform = `translate(-50%,-50%) scale(${active ? 1.35 : 1})`;
    dot.style.background = t ? t.color : 'rgba(255,255,255,0.9)';
  }

  return { targets, set, dispose: () => el.remove() };
}
