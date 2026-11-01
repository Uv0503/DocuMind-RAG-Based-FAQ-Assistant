"""Small, reviewable held-out document QA set. Labels are authored before retrieval.

The three existing company-policy documents in evaluation/documents/ are development
material. These eight unrelated public-service documents form the held-out set.
Every evidence quote below must occur verbatim in its named source document.
"""

HELDOUT = [
    {
        "source": "east_library.txt", "topic": "library", "text": """# East Borough Library
East Borough Library serves residents through a central branch and two neighborhood desks. Staff can explain loan rules at the counter, but they cannot override a hold placed by another reader. The catalog displays the due date for each item, and borrowers remain responsible for checking their account even when an email reminder is delayed.

## Print loans
Adult print books are due after 21 calendar days. A book may be renewed once if another reader has not placed a hold on it. Reference books stay inside the library, while children's picture books follow the same 21-day period as adult books. Returns placed in the outdoor slot after closing are checked in on the next open day.

## Holds and replacement
A requested book remains on the pickup shelf for five open days after the notification is sent. The library does not charge daily overdue fines for print books. An item still missing 45 days after its due date is billed at its replacement price, although the bill is removed if the original item is returned undamaged before a replacement is purchased.

## Digital collection and hours
Electronic books expire automatically after 14 days and cannot be returned through the outdoor slot. The central branch opens on Saturday from 10 a.m. to 4 p.m. and is closed on Sunday. Neighborhood desks have different hours, so readers should check the branch notice before traveling. Digital borrowing does not change a reader's print-book limit.
""", "questions": [
            ("How long can someone keep an adult print book from East Borough?", ["Adult print books are due after 21 calendar days."], "21 calendar days"),
            ("Can an East Borough book be extended when someone else is waiting for it?", ["A book may be renewed once if another reader has not placed a hold on it."], "No"),
            ("How many open days will East Borough keep a reserved title for pickup?", ["A requested book remains on the pickup shelf for five open days after the notification is sent."], "Five open days"),
            ("What happens to an East Borough print book that is still missing well after its due date?", ["An item still missing 45 days after its due date is billed at its replacement price, although the bill is removed if the original item is returned undamaged before a replacement is purchased."], "Replacement bill after 45 days"),
        ]
    },
    {
        "source": "west_library.txt", "topic": "library", "text": """# Westhaven Library Service
Westhaven issues one library card to each resident. Circulation staff review requests in the order received, including requests placed online. The city publishes changes to opening hours on its notice board; a cardholder should not assume East Borough's rules apply here because the systems are operated separately.

## Borrowing and renewals
Most circulating books may be kept for 28 calendar days. A loan can be renewed twice when there is no waiting list for the title. Popular new releases have a shorter loan period shown on their checkout slip. Borrowers may hold up to ten physical items at one time, but digital titles do not count toward that physical-item limit.

## Overdue items and holds
Overdue physical books incur a charge of 25 cents per day, capped at five dollars per item. A reserved item is held for seven open days at the chosen branch. The library sends a notice when it arrives, but a notification failure does not extend the pickup period. Accounts with unpaid replacement bills cannot borrow more physical books until the bill is resolved.

## Online material and weekend service
Digital audiobooks remain available for 21 days and then disappear from the app automatically. The main Westhaven branch opens on Sunday from 1 p.m. to 5 p.m. and closes on Monday. Staff at the main branch can help transfer a hold to a smaller desk before it becomes ready for pickup.
""", "questions": [
            ("What is the normal book loan period at Westhaven?", ["Most circulating books may be kept for 28 calendar days."], "28 calendar days"),
            ("What is the maximum late charge for one Westhaven physical book?", ["Overdue physical books incur a charge of 25 cents per day, capped at five dollars per item."], "$5"),
            ("Does Westhaven offer Sunday hours at its main branch?", ["The main Westhaven branch opens on Sunday from 1 p.m. to 5 p.m. and closes on Monday."], "Yes, 1 p.m. to 5 p.m."),
        ]
    },
    {
        "source": "harbor_rail.txt", "topic": "transport", "text": """# Harbor Rail Passenger Guide
Harbor Rail connects the central station with coastal towns. Passengers must keep a valid ticket available during the trip, including when traveling with a bicycle. A service alert may change a platform but does not by itself cancel a ticket or create a right to a cash refund.

## Changes and refunds
An unused flexible ticket may be refunded before its scheduled departure, minus a four-dollar processing fee. Once the scheduled train has departed, an unused ticket is not refundable. A passenger whose train is canceled by Harbor Rail may choose a full refund or a seat on the next available service, regardless of the ordinary processing fee.

## Bicycles and luggage
Folding bicycles travel free when folded and stored in the luggage area. A full-size bicycle requires a five-dollar bicycle reservation made before boarding. Passengers may bring two ordinary bags without charge; extra bags are accepted only when space is available and must be tagged at the station desk.

## Assistance and missed connections
Travelers who need boarding assistance should request it at least 24 hours before departure. If a Harbor Rail delay causes a missed connection on the same ticket, station staff will arrange a later train without another fare. The operator does not cover a separately booked hotel or flight after a missed connection.
""", "questions": [
            ("Can a passenger get money back for an unused Harbor Rail flexible ticket before leaving?", ["An unused flexible ticket may be refunded before its scheduled departure, minus a four-dollar processing fee."], "Yes, minus $4"),
            ("What does Harbor Rail charge for a full-size bicycle?", ["A full-size bicycle requires a five-dollar bicycle reservation made before boarding."], "$5 reservation"),
            ("If Harbor Rail cancels a train, what choices does the passenger have?", ["A passenger whose train is canceled by Harbor Rail may choose a full refund or a seat on the next available service, regardless of the ordinary processing fee."], "Full refund or next service"),
        ]
    },
    {
        "source": "bay_ferry.txt", "topic": "transport", "text": """# Bay Ferry Travel Terms
Bay Ferry operates passenger boats across the estuary. Sailings can be affected by visibility and strong winds. Staff announce delays at the terminal and on the booking page, but passengers should keep their booking reference until their journey has finished.

## Cancellations by passengers
A passenger may cancel a standard booking up to two hours before sailing and receive the fare less a six-dollar cancellation fee. A request made within two hours of sailing receives travel credit rather than a cash refund. The travel credit expires after six months and can be used on any published Bay Ferry crossing.

## Operator cancellations
When Bay Ferry cancels a sailing, the passenger may take the next available boat or request a full refund to the original payment method. Severe-weather cancellations are treated the same way as mechanical cancellations for this purpose. The company cannot reimburse a separate rail or airline ticket bought for a later connection.

## Bikes and accessibility
An ordinary bicycle travels free if a bicycle space was reserved when the passenger booked. The deck crew can provide a boarding ramp, but passengers needing ramp assistance should notify the terminal at least two hours before sailing. Oversized cargo bicycles are not carried on the passenger service.
""", "questions": [
            ("How late can I cancel a Bay Ferry booking for a cash refund?", ["A passenger may cancel a standard booking up to two hours before sailing and receive the fare less a six-dollar cancellation fee."], "Up to two hours before sailing, less $6"),
            ("What happens if Bay Ferry cancels because of bad weather?", ["When Bay Ferry cancels a sailing, the passenger may take the next available boat or request a full refund to the original payment method.", "Severe-weather cancellations are treated the same way as mechanical cancellations for this purpose."], "Next boat or full refund"),
            ("Is an ordinary bicycle charged an additional fare on Bay Ferry?", ["An ordinary bicycle travels free if a bicycle space was reserved when the passenger booked."], "No, with a reservation"),
        ]
    },
    {
        "source": "oakview_housing.txt", "topic": "housing", "text": """# Oakview Residence Handbook
Oakview is a student residence with shared kitchens and study rooms. Residents receive a room inventory on arrival and should report missing items promptly. The front desk can accept a repair request, but it cannot grant an exception to the guest or fire rules.

## Guests and quiet hours
An overnight guest may stay for no more than two nights in any seven-day period. The host must register the guest at the front desk before 10 p.m. Guests are never issued a resident key, and the host remains responsible for their conduct. Quiet hours run from 10 p.m. until 7 a.m. every day, including weekends and examination periods.

## Repairs and safety
Residents should report a leaking pipe to the emergency maintenance number immediately rather than waiting for the next routine inspection. Routine repair requests are reviewed within two working days. Cooking appliances may be used only in the shared kitchen; candles and portable heaters are prohibited in bedrooms.

## Leaving the residence
Before moving out, a resident must clear personal belongings and return the room key by noon on the final day. A cleaning charge may apply if the room is left with rubbish or damaged furniture. The inventory signed at move-in is used when the room is checked after departure.
""", "questions": [
            ("How many nights may an Oakview resident host an overnight visitor in a week?", ["An overnight guest may stay for no more than two nights in any seven-day period."], "Two nights"),
            ("When does Oakview's daily quiet period begin?", ["Quiet hours run from 10 p.m. until 7 a.m. every day, including weekends and examination periods."], "10 p.m."),
            ("Should someone at Oakview wait for an inspection before reporting a leaking pipe?", ["Residents should report a leaking pipe to the emergency maintenance number immediately rather than waiting for the next routine inspection."], "No, report immediately"),
        ]
    },
    {
        "source": "pinecrest_housing.txt", "topic": "housing", "text": """# Pinecrest Graduate Housing Rules
Pinecrest houses postgraduate students in self-contained apartments. The housing office publishes tenancy notices to each resident account. A resident must update emergency contact details after any change and may not transfer an apartment key to a visitor.

## Visitors and common areas
An overnight visitor may stay for up to three consecutive nights, but no more than six nights in one calendar month. Visitors must leave their name with the building desk when they arrive. Social gatherings in common rooms require a booking when more than eight people will attend. Quiet hours start at 11 p.m. and end at 8 a.m.

## Maintenance and utilities
Nonurgent maintenance requests are acknowledged within three working days. Residents should use the emergency line for a loss of heat in winter or a water leak. Building staff may enter for an emergency, but routine inspections are announced at least 48 hours in advance.

## Checkout
The housing office asks residents to give 30 days' notice before ending a month-to-month tenancy. The apartment must be cleared and keys returned by 5 p.m. on the final date. Deposits are assessed after the inventory and any outstanding utility charges are reviewed.
""", "questions": [
            ("What is Pinecrest's monthly ceiling on overnight guest stays?", ["An overnight visitor may stay for up to three consecutive nights, but no more than six nights in one calendar month."], "Six nights per month"),
            ("How far ahead must Pinecrest announce a routine apartment inspection?", ["Building staff may enter for an emergency, but routine inspections are announced at least 48 hours in advance."], "48 hours"),
            ("How much notice is requested before ending a Pinecrest month-to-month tenancy?", ["The housing office asks residents to give 30 days' notice before ending a month-to-month tenancy."], "30 days"),
        ]
    },
    {
        "source": "museum_care.txt", "topic": "museum", "text": """# Meridian Museum Collections Manual
Meridian Museum holds paintings, textiles, and archival photographs. The collection team records the location of each object before it moves between storage, conservation, and exhibition. Loan agreements are signed by the registrar, not by gallery attendants.

## Gallery environment
The painting gallery targets relative humidity between 45 and 55 percent. Staff log humidity readings every morning before visitors enter. A reading outside the target band triggers an alert to the conservator, who decides whether an object must be moved or the environmental equipment adjusted.

## Handling and photography
Unframed photographs must be handled with clean nitrile gloves and supported by a rigid board. Visitors may photograph permanent displays without flash, but flash photography is prohibited in every gallery. Tripods require written permission because they can obstruct an evacuation route.

## Outgoing loans
A borrowing museum must provide a climate report and proof of insurance before a loan is approved. The registrar prepares the condition report immediately before packing and again when the object returns. The borrowing venue pays for the approved art transport company, while Meridian retains final authority over packing materials.

## Records
Condition reports and transport records are kept for ten years after a loan closes. Public exhibition labels may summarize an object's history, but private donor addresses are excluded from those labels and from press materials.
""", "questions": [
            ("What humidity range does Meridian target for paintings?", ["The painting gallery targets relative humidity between 45 and 55 percent."], "45 to 55 percent"),
            ("May visitors use a flash when photographing a Meridian display?", ["Visitors may photograph permanent displays without flash, but flash photography is prohibited in every gallery."], "No"),
            ("What paperwork must a borrowing museum provide before Meridian approves a loan?", ["A borrowing museum must provide a climate report and proof of insurance before a loan is approved."], "Climate report and insurance proof"),
        ]
    },
    {
        "source": "river_monitoring.txt", "topic": "environment", "text": """# Riverbend Watershed Field Protocol
The Riverbend team collects water-quality observations at upstream and downstream stations. Each sample bottle carries a station name, date, and collector initials. Field staff write down weather and recent flow changes because heavy rain can alter readings even when the sampling instruments are working properly.

## Routine collection
Teams collect routine water samples every second Tuesday during the warm season. After a rainfall event exceeding 30 millimeters in one day, an additional sample is taken within 24 hours if access is safe. The extra sample does not replace the next routine visit.

## Instrument checks
The dissolved-oxygen meter is calibrated at the beginning of each field day using the manufacturer's procedure. If a calibration check fails, staff mark the reading invalid, use a backup meter if available, and record the failure in the field log. They must not copy a value from the prior visit into the current report.

## Reporting and records
An unexpected chemical reading is reported to the project lead before the team leaves the site. The lead decides whether a confirmation sample is required and informs the relevant authority when the threshold in the response plan is met. Raw field sheets and calibration logs are retained for five years after the annual report is issued.

## Public communication
The annual report distinguishes routine observations from event-driven samples. A public summary may explain trends but must not present an invalid meter reading as a verified result.
""", "questions": [
            ("How often does Riverbend do its scheduled warm-season sampling?", ["Teams collect routine water samples every second Tuesday during the warm season."], "Every second Tuesday"),
            ("What does the Riverbend team do when its oxygen meter fails calibration?", ["If a calibration check fails, staff mark the reading invalid, use a backup meter if available, and record the failure in the field log."], "Mark invalid, use backup if available, log failure"),
            ("Does a post-storm Riverbend sample cancel the next scheduled visit?", ["After a rainfall event exceeding 30 millimeters in one day, an additional sample is taken within 24 hours if access is safe.", "The extra sample does not replace the next routine visit."], "No"),
        ]
    },
]

UNANSWERABLE = [
    "At what time does East Borough Library open on Tuesday?",
    "How much does Harbor Rail pay for meals during a delay?",
    "What is the maximum security deposit at Pinecrest housing?",
    "Which chemical threshold triggers a Riverbend authority notification?",
]
