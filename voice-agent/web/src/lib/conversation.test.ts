import { expect, test } from 'bun:test'

import { isFinishedTurn, normalizeTranscript, normalizeTranscriptSpacing } from './conversation'

test('normalizeTranscriptSpacing inserts spaces and collapses whitespace', () => {
  expect(normalizeTranscriptSpacing('Hello.World,now  ok')).toBe('Hello. World, now ok')
})

test("normalizeTranscript remaps uid '0' to the local uid and normalizes text", () => {
  const out = normalizeTranscript(
    [
      { uid: '0', text: 'Hi.There', turn_id: '1', status: 0 },
      { uid: '42', text: 'ok', turn_id: '2', status: 0 },
      // biome-ignore lint/suspicious/noExplicitAny: minimal test fixtures
    ] as any,
    'local-9',
  )
  expect(out[0].uid).toBe('local-9')
  expect(out[0].text).toBe('Hi. There')
  expect(out[1].uid).toBe('42')
})

test('normalizeTranscript drops the silence-check control marker', () => {
  const out = normalizeTranscript(
    [
      { uid: '42', text: '[[ithink-silence-check]]', turn_id: '1', status: 1 },
      { uid: '42', text: '  [[ithink-silence-check]]  ', turn_id: '2', status: 1 },
      { uid: '42', text: 'real speech', turn_id: '3', status: 1 },
      // biome-ignore lint/suspicious/noExplicitAny: minimal test fixtures
    ] as any,
    'local-9',
  )
  expect(out).toHaveLength(1)
  expect(out[0].text).toBe('real speech')
})

test('isFinishedTurn only accepts END (1), not IN_PROGRESS (0) or INTERRUPTED (2)', () => {
  const item = (status: number) =>
    // biome-ignore lint/suspicious/noExplicitAny: minimal test fixture
    ({ turn_id: 1, uid: 9, text: 'hi', status, createdAt: 0 }) as any
  expect(isFinishedTurn(item(0))).toBe(false)
  expect(isFinishedTurn(item(1))).toBe(true)
  expect(isFinishedTurn(item(2))).toBe(false)
})
