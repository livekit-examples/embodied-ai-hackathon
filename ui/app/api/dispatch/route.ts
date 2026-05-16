import { NextResponse } from 'next/server';
import { AgentDispatchClient, RoomServiceClient } from 'livekit-server-sdk';

// NOTE: you are expected to define the following environment variables in `.env.local`:
const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const LIVEKIT_URL = process.env.LIVEKIT_URL;
const LIVEKIT_ROOM = process.env.LIVEKIT_ROOM;

// Dispatch name registered by the Python orchestrator via
// `@server.rtc_session(agent_name="orchestrator")`.
const AGENT_NAME = 'orchestrator';

export const revalidate = 0;

export async function POST() {
  if (process.env.NODE_ENV !== 'development') {
    throw new Error(
      'THIS API ROUTE IS INSECURE. DO NOT USE THIS ROUTE IN PRODUCTION WITHOUT AN AUTHENTICATION LAYER.'
    );
  }

  try {
    if (LIVEKIT_URL === undefined) {
      throw new Error('LIVEKIT_URL is not defined');
    }
    if (API_KEY === undefined) {
      throw new Error('LIVEKIT_API_KEY is not defined');
    }
    if (API_SECRET === undefined) {
      throw new Error('LIVEKIT_API_SECRET is not defined');
    }
    if (LIVEKIT_ROOM === undefined) {
      throw new Error('LIVEKIT_ROOM is not defined');
    }

    const roomName = LIVEKIT_ROOM;

    // Ensure the room exists before touching agent-dispatch APIs — those
    // 404 on rooms that haven't been created yet.
    const roomClient = new RoomServiceClient(LIVEKIT_URL, API_KEY, API_SECRET);
    const rooms = await roomClient.listRooms([roomName]);
    if (rooms.length === 0) {
      console.log(`[dispatch] room "${roomName}" doesn't exist yet, creating`);
      await roomClient.createRoom({ name: roomName });
    }

    // Clear all prior dispatches and create a fresh one so the user always
    // lands in a room with exactly one freshly-dispatched agent.
    const dispatchClient = new AgentDispatchClient(LIVEKIT_URL, API_KEY, API_SECRET);
    const existing = await dispatchClient.listDispatch(roomName);
    console.log(
      `[dispatch] room "${roomName}": found ${existing.length} existing dispatch(es), clearing all`
    );
    console.log(existing);
    await Promise.all(
      existing.map(async (d) => {
        console.log(`[dispatch]   deleting dispatch id=${d.id} agentName=${d.agentName}`);
        await dispatchClient.deleteDispatch(d.id, roomName);
      })
    );
    console.log(`[dispatch] creating fresh "${AGENT_NAME}" dispatch in "${roomName}"`);
    const created = await dispatchClient.createDispatch(roomName, AGENT_NAME);
    console.log(`[dispatch]   created dispatch id=${created.id}`);

    return NextResponse.json({ roomName, dispatchId: created.id });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);
      return new NextResponse(error.message, { status: 500 });
    }
  }
}
