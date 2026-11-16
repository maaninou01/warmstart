# "Personal?" test set

Customer messages for 8 fictional businesses, each labelled personal or general. It measures what the GrailQA sets can't: whether the "personal?" check lets a personal message through. GrailQA has no personal questions, so on it any looser bar looks free.

- **Personal (`yes`):** a correct reply depends on records the business holds about one customer, or on acting on them (orders, bookings, account, bills, usage), whether the sender's or another person's.
- **General (`no`):** one correct reply works for every customer who sends that exact message, even when it says "my" or "I".
- **`unsure`:** reviewers would split, for example "can i cancel". These are left out of the tuning and reported on their own.

| Split | Businesses | Personal | General | Unsure |
|---|---|---|---|---|
| tune | Northwind Outfitters (clothes shop), Brightline Mobile, Skyhop (airline), Pulse Gyms | 483 | 361 | 24 |
| test | Voltcart (electronics shop), Hearth Energy, Ledgerly (bank app), Forkful (food delivery) | 505 | 331 | 23 |

## How it was built

1. **`personal_set.py`:** Claude Haiku 5.5 wrote 20 messages per business and kind. There are 11 kinds:
   - personal: says "my", never says "my", 1–4 words, about another person, mixed with a general question, or claims to be general;
   - general: rules, says "my", 1–4 words, general knowledge, badly written.

   After removing duplicates, 1,727 messages remained. A second Haiku call checked each label.
2. **Blind review (`review.csv`, stage `before scoring`):** the check disagreed with the writer on 296 messages (17%). A reviewer read all 296, plus 60 random messages where they agreed, seeing only the business and the message.
   - On the 296 disagreements, the reviewer sided with the check 215 times, with the writer 42 times, and was unsure 39 times. Most were the writer's mistakes, for example "bill too high" written as general.
   - On the 60 agreements, it changed 3 and was unsure of 2. So about 5% of the unreviewed labels may be wrong (95% range 4–18%).
3. **Review after scoring (stage `after scoring`):** 16 more messages went to the same blind review, chosen because Jev got them wrong. They were 12 tuning messages labelled personal that scored under 0.5, and the 4 held-out personal messages that the tuned bar let through. Because Jev chose them, `results/personal.json` reports the held-out numbers both with and without this review.

## Known limits

- **A model wrote it, not customers.** A model from the same family as the checker wrote the messages, so they may be easier than real traffic.
- **Labels follow one definition.** For example, action requests like "lost my card" count as personal. Another business may draw the line elsewhere.
- **The held-out businesses aren't a clean test.** They decide whether a tuned bar may replace the strict one, so they act as a validation set.
