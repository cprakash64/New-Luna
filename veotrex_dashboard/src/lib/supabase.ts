import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!url || !anonKey) {
  throw new Error(
    "Missing VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY. Copy .env.example to .env.local and fill them in.",
  );
}

export const supabase = createClient(url, anonKey, {
  auth: { persistSession: false },
});

export type CameraConfigRow = {
  camera_id: string;
  roi_polygon: [number, number][];
  active_rules: string[];
  updated_at?: string;
};
