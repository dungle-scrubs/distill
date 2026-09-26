# An X post's URL id names the source

Distill gains a second remote **source** kind: one post on x.com (or
twitter.com) that carries a video. The identity decision is the whole of the
design, and it differs from YouTube's in one load-bearing way: a run keys the
**bundle** on the **status id read from the URL**, never on the media id
yt-dlp resolves.

## Why not the resolved id

For YouTube, the id in the URL is the id yt-dlp resolves - the extractor's own
regex guarantees it, which is what makes `youtube_fast_path_video_id` sound.
The X extractor makes no such promise. Its own test corpus carries posts where
the resolved `id` differs from the URL's `display_id`: a quote tweet, asked for
by the quoting post's id, resolves to the quoted media's id; a unified card
resolves to the card's media. Keying on the resolved id would therefore make
every cache lookup a question only yt-dlp can answer - and reverse the
cache-before-capability property (R-49) for this kind: a run serving an
**active generation** would need the acquisition tool installed to find the
bundle it does not need to acquire.

Keying on the URL's own status id keeps that property. The id is validated by
shape (`[0-9]+`), normalized (query string and fragment stripped, host
canonicalized), and hashed in a distinct domain
(`sha256("x:" + status_id)`) so an X id and a YouTube id that spelled the same
string could never share a **bundle key**. Metadata from yt-dlp describes the
post (title, uploader, text, upload date) but never participates in identity.

## Consequences

- A cache hit for an X URL costs one URL parse. yt-dlp is needed only to
  *produce* a generation, exactly as R-49 states.
- Two URLs that surface the same underlying media - a post, and the post that
  quoted it - produce two bundles. The duplication is bounded by how often
  people quote-tweet videos, and resolving it would cost the property above.
  Deliberate cache decision, taken here rather than left to emerge.
- The extractor accepts `/video/N` and `/photo/N` URL suffixes. Distill
  refuses them: the convention below names one video per post, and a suffix
  naming a *different* video keyed under the same id would serve the wrong
  bundle - the exact hazard the YouTube fast path declines `list=` for.

## One post, one video

A post can carry several videos, and the extractor returns them as a playlist;
`--no-playlist` does not collapse it (the extractor's `_yes_playlist` decides
before the flag is consulted). The convention: a run processes the **first
video** of the post. Every download invocation carries `--playlist-items 1`,
so exactly one media file is staged; metadata reads the first document
`--dump-json` emits and records a **degradation warning** when the document
count shows the post carried more. The convention is deterministic, which is
what makes URL-id keying sound: the same URL always names the same media.

## Authentication

X serves its GraphQL response to unauthenticated (guest) sessions with the
video stripped, so in practice a post's video is only extractable with the
operator's own session - the recurring "No video could be found in this tweet"
of yt-dlp's issue tracker is this, not a missing extractor. Distill therefore
carries two authentication options, `cookies` and `cookies_from_browser`
(`--cookies` / `--cookies-from-browser`), passed through to every yt-dlp
invocation the run makes. They are **machine-local claims**, exactly as the
vision endpoint's address and credential are: they reach the tool's argv and
nothing else. Their options are `cache_key=False`, so they enter neither the
**options hash** nor any **manifest** - which jar authenticated a run does not
describe what the run produced (ADR-0004). Guest access is what a run without
these options does; when X allows it, the run works unauthenticated, and when
it does not, the run fails with the ordinary acquisition error naming the
posts it could not read.
