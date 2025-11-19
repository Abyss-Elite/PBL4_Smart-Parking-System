"use client";

import SlotCard from "./SlotCard";

export default function SlotList({ slots, selectedSlot, cars, booked, onSelectSlot }) {
  const getSlotBookedTimes = (slot) => {
    const initial = booked[slot] || [];
    const fromCars = cars
      .filter((c) => c.slot === slot)
      .map((c) => ({ start: c.start, end: c.end }));
    return [...initial, ...fromCars];
  };

  return (
    <div className="space-y-4">
      <div className="grid max-h-[85vh] grid-cols-2 gap-3 overflow-y-auto pr-2">
        {slots.map((s) => {
          const bookedTimes = getSlotBookedTimes(s);
          return (
            <SlotCard
              key={s}
              slot={s}
              bookedTimes={bookedTimes}
              selected={selectedSlot === s}
              onClick={() => onSelectSlot(s)}
            />
          );
        })}
      </div>
    </div>
  );
}
