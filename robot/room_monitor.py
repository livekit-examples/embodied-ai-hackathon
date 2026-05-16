"""LiveKit room monitor: connect with creds from .env and log every event.

Used to debug why a robot is or isn't publishing video tracks. Connects as a
passive subscriber to the same room as `robot.py`, then prints both the
initial participant/track inventory and every subsequent room event.

Run with:
    uv run room_monitor.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from livekit import rtc

from _common import load_env, mint_token, required_env

IDENTITY = "room-monitor"

logger = logging.getLogger("room_monitor")


def _fmt_participant(p: rtc.Participant | None) -> str:
    if p is None:
        return "<none>"
    return f"{p.identity!r}(sid={p.sid})"


def _fmt_publication(pub: rtc.TrackPublication | None) -> str:
    if pub is None:
        return "<none>"
    return (
        f"{pub.kind.name if hasattr(pub.kind, 'name') else pub.kind}"
        f"/{pub.source.name if hasattr(pub.source, 'name') else pub.source}"
        f" name={pub.name!r} sid={pub.sid} muted={pub.muted}"
        f" subscribed={getattr(pub, 'subscribed', '?')}"
        f" mime={getattr(pub, 'mime_type', '?')!r}"
    )


def _fmt_track(track: rtc.Track | None) -> str:
    if track is None:
        return "<none>"
    return (
        f"kind={track.kind.name if hasattr(track.kind, 'name') else track.kind}"
        f" sid={track.sid} muted={track.muted}"
    )


def _log_initial_state(room: rtc.Room) -> None:
    logger.info(
        "[monitor] connected: room=%r sid=%s local=%s",
        room.name,
        getattr(room, "sid", "?"),
        _fmt_participant(room.local_participant),
    )
    remotes = list(room.remote_participants.values())
    logger.info("[monitor] remote participants at connect: %d", len(remotes))
    for p in remotes:
        logger.info("[monitor]   - %s", _fmt_participant(p))
        for pub in p.track_publications.values():
            logger.info("[monitor]       pub: %s", _fmt_publication(pub))


def _attach_event_loggers(room: rtc.Room) -> None:
    """Wire up a callback for every documented Room event type.

    We register one handler per event name so each line shows the event name
    plus a hand-formatted view of its args.
    """

    @room.on("participant_connected")
    def _(p: rtc.RemoteParticipant) -> None:
        logger.info("[evt] participant_connected %s", _fmt_participant(p))

    @room.on("participant_disconnected")
    def _(p: rtc.RemoteParticipant) -> None:
        logger.info("[evt] participant_disconnected %s", _fmt_participant(p))

    @room.on("participant_active")
    def _(p: rtc.RemoteParticipant) -> None:
        logger.info("[evt] participant_active %s", _fmt_participant(p))

    @room.on("local_track_published")
    def _(pub: rtc.LocalTrackPublication, track: rtc.Track) -> None:
        logger.info(
            "[evt] local_track_published pub=(%s) track=(%s)",
            _fmt_publication(pub),
            _fmt_track(track),
        )

    @room.on("local_track_unpublished")
    def _(pub: rtc.LocalTrackPublication) -> None:
        logger.info("[evt] local_track_unpublished pub=(%s)", _fmt_publication(pub))

    @room.on("local_track_republished")
    def _(pub: rtc.LocalTrackPublication, previous_sid: str) -> None:
        logger.info(
            "[evt] local_track_republished pub=(%s) previous_sid=%s",
            _fmt_publication(pub),
            previous_sid,
        )

    @room.on("local_track_subscribed")
    def _(track: rtc.Track) -> None:
        logger.info("[evt] local_track_subscribed track=(%s)", _fmt_track(track))

    @room.on("track_published")
    def _(pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
        logger.info(
            "[evt] track_published from=%s pub=(%s)",
            _fmt_participant(p),
            _fmt_publication(pub),
        )

    @room.on("track_unpublished")
    def _(pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
        logger.info(
            "[evt] track_unpublished from=%s pub=(%s)",
            _fmt_participant(p),
            _fmt_publication(pub),
        )

    @room.on("track_subscribed")
    def _(
        track: rtc.Track,
        pub: rtc.RemoteTrackPublication,
        p: rtc.RemoteParticipant,
    ) -> None:
        logger.info(
            "[evt] track_subscribed from=%s pub=(%s) track=(%s)",
            _fmt_participant(p),
            _fmt_publication(pub),
            _fmt_track(track),
        )

    @room.on("track_unsubscribed")
    def _(
        track: rtc.Track,
        pub: rtc.RemoteTrackPublication,
        p: rtc.RemoteParticipant,
    ) -> None:
        logger.info(
            "[evt] track_unsubscribed from=%s pub=(%s) track=(%s)",
            _fmt_participant(p),
            _fmt_publication(pub),
            _fmt_track(track),
        )

    @room.on("track_subscription_failed")
    def _(p: rtc.RemoteParticipant, sid: str, name: str) -> None:
        logger.warning(
            "[evt] track_subscription_failed from=%s sid=%s name=%r",
            _fmt_participant(p),
            sid,
            name,
        )

    @room.on("track_muted")
    def _(p: rtc.Participant, pub: rtc.TrackPublication) -> None:
        logger.info(
            "[evt] track_muted from=%s pub=(%s)",
            _fmt_participant(p),
            _fmt_publication(pub),
        )

    @room.on("track_unmuted")
    def _(p: rtc.Participant, pub: rtc.TrackPublication) -> None:
        logger.info(
            "[evt] track_unmuted from=%s pub=(%s)",
            _fmt_participant(p),
            _fmt_publication(pub),
        )

    @room.on("active_speakers_changed")
    def _(speakers: list[rtc.Participant]) -> None:
        logger.info(
            "[evt] active_speakers_changed speakers=%s",
            [_fmt_participant(s) for s in speakers],
        )

    @room.on("room_metadata_changed")
    def _(old: str, new: str) -> None:
        logger.info("[evt] room_metadata_changed old=%r new=%r", old, new)

    @room.on("participant_metadata_changed")
    def _(p: rtc.Participant, old: str, new: str) -> None:
        logger.info(
            "[evt] participant_metadata_changed from=%s old=%r new=%r",
            _fmt_participant(p),
            old,
            new,
        )

    @room.on("participant_name_changed")
    def _(p: rtc.Participant, old: str, new: str) -> None:
        logger.info(
            "[evt] participant_name_changed from=%s old=%r new=%r",
            _fmt_participant(p),
            old,
            new,
        )

    @room.on("participant_attributes_changed")
    def _(changed: dict[str, str], p: rtc.Participant) -> None:
        logger.info(
            "[evt] participant_attributes_changed from=%s changed=%r",
            _fmt_participant(p),
            changed,
        )

    @room.on("connection_quality_changed")
    def _(p: rtc.Participant, quality: Any) -> None:
        logger.info(
            "[evt] connection_quality_changed from=%s quality=%r",
            _fmt_participant(p),
            quality,
        )

    @room.on("participant_encryption_status_changed")
    def _(p: rtc.Participant, encrypted: bool) -> None:
        logger.info(
            "[evt] participant_encryption_status_changed from=%s encrypted=%s",
            _fmt_participant(p),
            encrypted,
        )

    @room.on("participant_permissions_changed")
    def _(p: rtc.Participant) -> None:
        logger.info(
            "[evt] participant_permissions_changed from=%s permissions=%r",
            _fmt_participant(p),
            getattr(p, "permissions", None),
        )

    @room.on("data_received")
    def _(packet: rtc.DataPacket) -> None:
        # Only log size + topic; payload bytes are noisy.
        data = getattr(packet, "data", b"") or b""
        logger.info(
            "[evt] data_received from=%s topic=%r kind=%r bytes=%d",
            _fmt_participant(getattr(packet, "participant", None)),
            getattr(packet, "topic", None),
            getattr(packet, "kind", None),
            len(data),
        )

    @room.on("sip_dtmf_received")
    def _(packet: Any) -> None:
        logger.info("[evt] sip_dtmf_received packet=%r", packet)

    @room.on("transcription_received")
    def _(segments: list[Any], p: rtc.Participant, pub: rtc.TrackPublication) -> None:
        logger.info(
            "[evt] transcription_received from=%s pub=(%s) segments=%d",
            _fmt_participant(p),
            _fmt_publication(pub),
            len(segments),
        )

    @room.on("e2ee_state_changed")
    def _(p: rtc.Participant, state: Any) -> None:
        logger.info(
            "[evt] e2ee_state_changed from=%s state=%r",
            _fmt_participant(p),
            state,
        )

    @room.on("connection_state_changed")
    def _(state: Any) -> None:
        logger.info("[evt] connection_state_changed state=%r", state)

    @room.on("connected")
    def _() -> None:
        logger.info("[evt] connected")

    @room.on("disconnected")
    def _(reason: Any = None) -> None:
        logger.info("[evt] disconnected reason=%r", reason)

    @room.on("reconnecting")
    def _() -> None:
        logger.info("[evt] reconnecting")

    @room.on("reconnected")
    def _() -> None:
        logger.info("[evt] reconnected")

    @room.on("room_updated")
    def _(*args: Any) -> None:
        logger.info("[evt] room_updated args=%r", args)

    @room.on("moved")
    def _(*args: Any) -> None:
        logger.info("[evt] moved args=%r", args)

    @room.on("token_refreshed")
    def _(*args: Any) -> None:
        logger.info("[evt] token_refreshed args=%r", args)

    @room.on("data_track_published")
    def _(track: Any) -> None:
        logger.info("[evt] data_track_published track=%r", track)

    @room.on("data_track_unpublished")
    def _(sid: str) -> None:
        logger.info("[evt] data_track_unpublished sid=%s", sid)


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env()

    url = required_env("LIVEKIT_URL")
    room_name = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room_name)

    room = rtc.Room()
    _attach_event_loggers(room)

    logger.info(
        "[monitor] connecting to %s as %r in room %r ...", url, IDENTITY, room_name
    )
    # auto_subscribe so we get track_subscribed for any track currently in the
    # room, which is what we actually care about when a robot's video is MIA.
    await room.connect(
        url,
        token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )
    _log_initial_state(room)

    try:
        # Sleep forever; events are delivered via callbacks.
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("[monitor] interrupted; disconnecting ...")
    finally:
        await room.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
