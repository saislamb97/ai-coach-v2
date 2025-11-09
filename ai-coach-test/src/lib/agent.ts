import { get, patch } from './http'
import { ENV } from './env'

export type AgentRead = {
  bot_id: string
  name: string
  glb_url?: string | null
}

export async function fetchAgent(): Promise<AgentRead> {
  return get(`/api/agents/${ENV.BOT_ID}/`)
}

export async function saveAgentGlb(glb_url: string) {
  return patch(`/api/agents/${ENV.BOT_ID}/`, { glb_url })
}
