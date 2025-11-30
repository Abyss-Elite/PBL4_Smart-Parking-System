import { Car } from "@/types/car";

export type BookingMode = "week" | "month";

export interface BookingRange {
  startTimeBooking: string;
  endTimeBooking: string;
}

type OldCar = {
  plate: string;
  mode: "week" | "month";
  weekStart?: string;
  month?: string;    
  slot: string;
};

export function getBookingRange(mode: BookingMode, date: string): BookingRange {
  const formatLocal = (d: Date) =>
    d.toLocaleString("sv-SE", { hour12: false }).replace(" ", "T");
  if (mode === "week") {
    const start = new Date(date);
    start.setHours(0, 0, 0, 0);

    const end = new Date(start);
    end.setDate(end.getDate() + 6);
    end.setHours(23, 59, 59, 999);

    return {
      startTimeBooking: formatLocal(start),
      endTimeBooking: formatLocal(end),
    };
  } else if (mode === "month") {
    if (date.length === 7) date = date + "-01";

    const start = new Date(date);
    start.setHours(0, 0, 0, 0);

    const end = new Date(start.getFullYear(), start.getMonth() + 1, 0);
    end.setHours(23, 59, 59, 999);

    return {
      startTimeBooking: formatLocal(start),
      endTimeBooking: formatLocal(end),
    };
  }

  throw new Error("mode phải là 'week' hoặc 'month'");
}

export function convertOldCarsToNew(oldCars: OldCar[]): Car[] {
  return oldCars.map((oldCar) => {
    const date = oldCar.mode === "week" ? oldCar.weekStart : oldCar.month;
    if (!date) throw new Error("OldCar thiếu thông tin ngày");

    const { startTimeBooking, endTimeBooking } = getBookingRange(oldCar.mode, date);

    return {
      licensePlate: oldCar.plate,
      dateBooking: date,
      startTimeBooking,
      endTimeBooking,
      spotName: oldCar.slot,
      mode: oldCar.mode
    };
  });
}