// typescript/m1/Practice/judge_card_practice.ts
/**
 * M1 Practice: Build a Judge Persona that scores you and renders a card.
 *
 * You answer an 8-question personality quiz. An agent running a judge
 * persona (pirate, ancient mummy, savage critic, or your own) scores your
 * answers, matches you to a real LangChain product, and renders a shareable
 * result card as ASCII art in your terminal.
 *
 * The quiz, the card renderer, the mock "post" tool, and the approval loop
 * are provided in judge_card_helpers.ts (same idea as models.ts: shared
 * setup you import). You don't need to open it to do this practice.
 *
 * YOUR TODOS: work top to bottom. The lesson page has the full walkthrough.
 *   TODO 1 (Lesson 1.4) Write your own judge persona.
 *   TODO 2 (Lesson 1.5) Finish scoreAndMatch. The script stops with a
 *          "TODO 2" message until this is done. Run it after this one.
 *   TODO 3 (Lesson 1.7) Add a second judge to JUDGES_TO_RUN. Run it again.
 *   TODO 4 (Lesson 1.8) Set INTERRUPT_ON so posting needs your approval.
 *          Run it again.
 *   TODO 5 (Lesson 1.3, optional) Try strongModel.
 *   TODO 6 (Lesson 1.6, stretch goal) Replace the placeholder product fact
 *          with a real one from the LangChain docs MCP server.
 *
 * RUN
 *   cd typescript && pnpm tsx m1/Practice/judge_card_practice.ts
 *
 * ════════════════════════════════════════════════════════════════════════
 *   SHARE IT: got a card you like? Screenshot it, tag @LangChain
 *   on X or LinkedIn, and show us your work!
 * ════════════════════════════════════════════════════════════════════════
 */

import { context, tool } from "langchain";
import { z } from "zod";

import {
  OUTPUT_DIR,
  PRODUCT_MATCHES,
  TOOL_SEQUENCE,
  TRAIT_AXES,
  postCard,
  renderCard,
  runJudge,
  runQuiz,
  stopForTodo,
  type TraitDelta,
} from "./judge_card_helpers.js";
import { model } from "../../models.js";

// ════════════════════════════════════════════════════════════════════════
// TODO 1 (Lesson 1.4, The System Prompt: Persona)
// Three example judges are already written below. Write a fourth,
// "your_persona", in a voice all your own: it's the judge that runs by
// default, so it's the card that gets posted.
//
// You only write the voice. TOOL_SEQUENCE (added to the end of every
// persona) already tells the agent what to do: score three traits, match a
// product, then render and post the card. Start with "You are <Name>, ...";
// the agent signs your card with that name.
// Make it genuinely rude / roast you (if you want).
// ════════════════════════════════════════════════════════════════════════

export const JUDGE_PERSONAS: Record<string, string> = {
  salty_pirate:
    context`
      You are Captain Hardcode, a swashbuckling pirate
      captain judging landlubbers' habits as a builder (developer) as if
      inspecting new crew for seaworthiness before a voyage. Speak in thick,
      theatrical pirate dialect at all times ("arrr," "ye scallywag," "shiver
      me timbers," "walk the plank") and never break character into plain
      modern speech, not even once. Treat every trait score like cargo being
      weighed and measured, threaten keelhauling or marooning for weak,
      wishy-washy answers, and promise a share of the plunder and a place among
      the crew for bold, decisive ones.` + TOOL_SEQUENCE,

  ancient_mummy:
    context`
      You are Nefer-Ka, a 3,000-year-old mummy torn from an
      eternal slumber for the sole, sacred purpose of judging this mortal's
      habits as a builder (developer). Never speak plainly: every verdict must
      sound like a proclamation carved into a tomb wall. Reach for archaic,
      regal diction ("hear me, mortal," "so speaks the tomb," "let it be
      written"), invoke a curse or blessing in EVERY verdict without exception
      (not only for mediocre answers), and treat this quiz with the utmost
      sacred solemnity even though the questions are mundane office trivia. If
      a sentence could be spoken by a calm HR consultant, it has failed you -
      rewrite it until it could only be spoken by something risen from a
      sarcophagus.` + TOOL_SEQUENCE,

  savage_critic:
    context`
      You are Vex, a personality-quiz judge with the
      withering, theatrical condescension of someone who has seen your type a
      thousand times and finds you aggressively, personally underwhelming every
      single time. Never answer in flat or neutral language: sigh audibly in
      text, lean hard into backhanded compliments ("oh, adorable, you actually
      tried"), and act like reviewing this quiz is a personal favor you're
      doing the user, one you deeply regret. Every verdict should read like an
      eye-roll delivered as a formal statement. Talk down to the user like
      they're a mildly disappointing intern who needs everything explained
      twice: address them with a pet name that is not a compliment ("sweetie,"
      "champ," "darling"), and treat every question you were asked as an
      obviously stupid one you're too tired to be surprised by anymore. If a
      sentence could plausibly be said by a mildly annoyed customer service
      rep, it isn't cutting enough yet; sharpen it until it sounds like Vex
      can barely be bothered to look up from whatever they were doing to
      deliver it. You are sharp, a little cruel, and allergic to participation
      trophies.` + TOOL_SEQUENCE,

  your_persona:
    context`
      TODO 1: replace this text with your own judge persona. Start with
      "You are <Name>, ..." and describe a voice that is completely
      different from the three judges above.` + TOOL_SEQUENCE,
};

// ════════════════════════════════════════════════════════════════════════
// TODO 2 (Lesson 1.5, Tools: Custom Tools)
// The tallying (scoring each answer, then clamping to 0-100) is done for
// you; read the comments to see how it works. Your job starts at the
// "TODO here" comment: turn the finished scores array into a matched product.
// Until this is done, the script stops with a "TODO 2" message.
// ════════════════════════════════════════════════════════════════════════

export const scoreAndMatch = tool(
  ({ answers }: { answers: TraitDelta[] }): { traitScores: number[]; product: string } => {
    // Each of the 3 trait scores (chaotic/organized, cautious/bold,
    // solo/collaborative) starts neutral, at 50.
    const scores = [50, 50, 50];
    // answers is an array of [delta1, delta2, delta3] tuples, one per
    // question. Add each delta onto its matching score.
    for (const deltaTuple of answers) {
      for (let i = 0; i < 3; i++) {
        scores[i] += deltaTuple[i];
      }
    }
    // A long run of the same answer could push a score past 0 or 100, so
    // clamp every score back into that range.
    const clamped = scores.map((score) => Math.max(0, Math.min(100, score)));

    // TODO here: clamped is finished. Use it to pick a matched product.
    // 1. Set axisIndex to the index (0, 1, or 2) of whichever score in
    //    clamped is furthest from 50, i.e. has the biggest Math.abs(score - 50).
    //    Hint: this is a "find the index of the biggest value" problem, e.g.
    //    [0, 1, 2].reduce((best, i) => (... bigger than best ? i : best)).
    // 2. TRAIT_AXES[axisIndex] is a [leftLabel, rightLabel] tuple, e.g.
    //    ["Chaotic", "Organized"]. Set direction to whichever label matches
    //    the side clamped[axisIndex] leans toward: the right label if
    //    clamped[axisIndex] >= 50, otherwise the left label.
    // 3. Set product to PRODUCT_MATCHES[direction.toLowerCase()], e.g.
    //    PRODUCT_MATCHES["chaotic"] -> "Fleet".
    // 4. Return { traitScores: clamped, product }.
    return stopForTodo("TODO 2: see the comments above");
  },
  {
    name: "score_and_match",
    description:
      "Tally the quiz answers into three 0-100 trait scores and pick a matching LangChain product. Call this first, with the exact answers list you were given.",
    schema: z.object({
      answers: z.array(z.tuple([z.number(), z.number(), z.number()])),
    }),
  }
);

// ════════════════════════════════════════════════════════════════════════
// TODO 3 (Lesson 1.7, Messages, Threads, and Checkpointers: Threads)
// Add one of the example judges' keys ("salty_pirate", "ancient_mummy", or
// "savage_critic") to this list; you don't need to write another persona.
// Each judge runs in its own thread, so you get one card per judge, all
// judging the same quiz answers.
// ════════════════════════════════════════════════════════════════════════

export const JUDGES_TO_RUN = ["your_persona"]; // TODO 3: e.g. ["your_persona", "ancient_mummy"]

// ════════════════════════════════════════════════════════════════════════
// TODO 4 (Lesson 1.8, Human-in-the-Loop: Decision Types)
// Right now post_card runs without asking. Set INTERRUPT_ON so the agent
// pauses before posting. When it pauses, you'll see a draft of the post and
// can approve, edit, or reject it.
// ════════════════════════════════════════════════════════════════════════

const INTERRUPT_ON = undefined; // TODO 4: e.g. { post_card: true }

// ════════════════════════════════════════════════════════════════════════
// TODO 5 (Lesson 1.3, Models, optional)
// Import strongModel (next to model in the import at the top of this file),
// set MODEL to it, and compare the comedic timing.
// ════════════════════════════════════════════════════════════════════════

const MODEL = model; // TODO 5 (optional): e.g. const MODEL = strongModel;

// ════════════════════════════════════════════════════════════════════════
// TODO 6 (Lesson 1.6, MCP: Connecting Agents to External Services)
// A stretch goal. Until you do it, this tool returns PLACEHOLDER_FACT, so
// the rest of the practice runs without it.
//
// scoreAndMatch (TODO 2) already picked your product; this tool only
// describes it with one real fact from the docs. No login, API key, or
// account needed: docs.langchain.com/mcp is a public server.
//
// Mirror m1.6_agent_mcp.ts:
//   1. Connect to https://docs.langchain.com/mcp with MultiServerMCPClient.
//   2. Filter its tools down to just "search_docs_by_lang_chain".
//   3. Spin up a tiny agent with that one tool and ask it to describe
//      `product` in ONE short factual sentence (under 25 words).
//   4. Return that sentence, stripped of extra whitespace.
//
// The tool's function is already `async`, so await the MCP/agent calls
// directly inside it.
//
// On any failure (no network, tool error), log the error and return
// PLACEHOLDER_FACT so the practice stays runnable. The log is how you'll
// know it failed: the agent gets the same placeholder text in both cases.
// ════════════════════════════════════════════════════════════════════════

export const PLACEHOLDER_FACT = "No docs fact available. Base the verdict on the trait scores instead.";

export const fetchProductFact = tool(
  // TODO 6: replace this function with the MCP lookup described above.
  async ({ product }: { product: string }): Promise<string> => {
    return PLACEHOLDER_FACT;
  },
  {
    name: "fetch_product_fact",
    description:
      "Look up one grounded, factual sentence about the LangChain product you were matched with. Call this right after score_and_match, passing in the product name it returned.",
    schema: z.object({ product: z.string() }),
  }
);

export function buildUserPrompt(answers: TraitDelta[]): string {
  return (
    "Here are my personality quiz answers as a list of " +
    "(chaotic/organized, cautious/bold, solo/collaborative) deltas, in " +
    `order: ${JSON.stringify(answers)}. Call score_and_match with this exact list, then ` +
    "fetch_product_fact with the product it returns, then render and " +
    "post my card."
  );
}

const answers = await runQuiz();
const userPrompt = buildUserPrompt(answers);
for (const judgeName of JUDGES_TO_RUN) {
  await runJudge(judgeName, {
    systemPrompt: JUDGE_PERSONAS[judgeName],
    userPrompt,
    tools: [scoreAndMatch, fetchProductFact, renderCard, postCard],
    model: MODEL,
    interruptOn: INTERRUPT_ON,
  });
}
console.log(`\nCards saved to ${OUTPUT_DIR}/`);
