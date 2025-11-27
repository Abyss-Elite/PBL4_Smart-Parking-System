"use client";
import SlotCard from "./SlotCard";

export default function SlotList({
  slots,
  selectedSlot,
  cars,
  booked,
  bookingMode,
  weekStart,
  month,
  onSelectSlot,
}) {
  const yourCarsForSlot = (slot) => cars.filter((c) => c.slot === slot);

  return (
    <div className="space-y-4">
      <div className="grid max-h-[85vh] grid-cols-2 gap-3 overflow-y-auto pr-2">
        {slots.map((s) => (
          <SlotCard
            key={s}
            slot={s}
            booked={booked[s] || { week: [], month: [] }}
            bookingMode={bookingMode}
            currentWeek={weekStart}
            currentMonth={month}
            userCars={yourCarsForSlot(s)}
            onSelectCar={onSelectSlot}
            selected={selectedSlot}
          />
        ))}
      </div>
    </div>
  );
}
