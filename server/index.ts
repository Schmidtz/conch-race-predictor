import express from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { calculateConchStats } from '../src/services/statsService';
import { runMonteCarloSimulation } from '../src/services/monteCarlo';
import { ALL_CONCHES, EMOTICON_CONFIGS } from '../src/data/defaultConches';
import type { RaceRecord, RaceParticipantInput, SimulationConfig, EmoticonConfig, EmoticonType } from '../src/types';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const dataDir = path.join(root, 'data');
const dbPath = path.join(dataDir, 'races.json');

const app = express();
app.use(express.json({ limit: '5mb' }));

let records: RaceRecord[] = [];
let writeQueue = Promise.resolve();

async function loadDb() {
  await fs.mkdir(dataDir, { recursive: true });
  try {
    records = JSON.parse(await fs.readFile(dbPath, 'utf8'));
    if (!Array.isArray(records)) records = [];
  } catch {
    records = [];
    await saveDb();
  }
}

function saveDb() {
  writeQueue = writeQueue.then(async () => {
    const tmp = `${dbPath}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(records, null, 2), 'utf8');
    await fs.rename(tmp, dbPath);
  });
  return writeQueue;
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, records: records.length, modelVersion: 'Hybrid DB → Bayesian features → Monte Carlo v3' });
});

app.get('/api/records', (_req, res) => {
  res.json({ records });
});

app.post('/api/records', async (req, res) => {
  const input = req.body as Omit<RaceRecord, 'id'>;
  if (!input || !Array.isArray(input.participants) || !input.winnerId) {
    return res.status(400).json({ error: 'Invalid race record.' });
  }
  const record: RaceRecord = {
    ...input,
    id: `race_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  };
  records = [record, ...records];
  await saveDb();
  res.status(201).json(record);
});

app.post('/api/records/bulk', async (req, res) => {
  const incoming = req.body?.records;
  if (!Array.isArray(incoming)) return res.status(400).json({ error: 'records must be an array.' });
  const normalized: RaceRecord[] = incoming.map((r: RaceRecord, i: number) => ({
    ...r,
    id: r.id || `race_${Date.now()}_${i}_${Math.random().toString(36).slice(2, 7)}`,
  }));
  records = normalized;
  await saveDb();
  res.json({ records });
});

app.delete('/api/records/:id', async (req, res) => {
  const before = records.length;
  records = records.filter((r) => r.id !== req.params.id);
  if (records.length === before) return res.status(404).json({ error: 'Record not found.' });
  await saveDb();
  res.status(204).end();
});

app.delete('/api/records', async (_req, res) => {
  records = [];
  await saveDb();
  res.status(204).end();
});

app.post('/api/predict', (req, res) => {
  const participants = req.body?.participants as RaceParticipantInput[];
  const config = req.body?.config as SimulationConfig | undefined;
  const emoticonConfigs = req.body?.emoticonConfigs as Record<EmoticonType, EmoticonConfig> | undefined;

  if (!Array.isArray(participants) || participants.length < 2) {
    return res.status(400).json({ error: 'At least two race participants are required.' });
  }

  const safeConfig: SimulationConfig = {
    iterations: Math.min(1_000_000, Math.max(5_000, Math.floor(config?.iterations ?? 100_000))),
    emoticonWeight: Number(config?.emoticonWeight ?? 1),
    historicalWeight: Number(config?.historicalWeight ?? 1),
    popularityWeight: Number(config?.popularityWeight ?? 0.25),
    randomVariance: Number(config?.randomVariance ?? 0.35),
    raceDistance: Number(config?.raceDistance ?? 100),
  };

  const stats = calculateConchStats(records, ALL_CONCHES);
  const output = runMonteCarloSimulation(
    participants,
    stats,
    emoticonConfigs || EMOTICON_CONFIGS,
    safeConfig,
    ALL_CONCHES,
  );

  // Server is allowed to run up to 1M simulations. The shared model is therefore
  // no longer limited by browser CPU/time.
  res.json(output);
});

const distDir = path.join(root, 'dist');
app.use(express.static(distDir));
app.get('*', async (_req, res) => {
  try {
    await fs.access(path.join(distDir, 'index.html'));
    res.sendFile(path.join(distDir, 'index.html'));
  } catch {
    res.status(404).send('Frontend build not found. Run npm run build first.');
  }
});

const port = Number(process.env.PORT || 8787);
loadDb().then(() => {
  app.listen(port, () => console.log(`Conch Race Predictor server listening on :${port}`));
});
