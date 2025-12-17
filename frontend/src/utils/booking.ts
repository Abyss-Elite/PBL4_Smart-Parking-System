import { Car } from "@/types/car";

export type BookingMode = "week" | "month";

export interface BookingRange {
  startTimeBooking: string;
  endTimeBooking: string;
}

type OldCar = {
  licensePlate: string;
  mode: "week" | "month";
  startTimeBooking?: string;
  endTimeBooking?: string;
  spotName: string;
  spotId: number;
};

// export function getBookingRange(mode: BookingMode, date: string): BookingRange {
//   const formatLocal = (d: Date) =>
//     d.toLocaleString("sv-SE", { hour12: false }).replace(" ", "T");
//   if (mode === "week") {
//     const start = new Date(date);
//     start.setHours(0, 0, 0, 0);

//     const end = new Date(start);
//     end.setDate(end.getDate() + 6);
//     end.setHours(23, 59, 59, 999);

//     return {
//       startTimeBooking: formatLocal(start),
//       endTimeBooking: formatLocal(end),
//     };
//   } else if (mode === "month") {
//     if (date.length === 7) date = date + "-01";

//     const start = new Date(date);
//     start.setHours(0, 0, 0, 0);

//     const end = new Date(start.getFullYear(), start.getMonth() + 1, 0);
//     end.setHours(23, 59, 59, 999);

//     return {
//       startTimeBooking: formatLocal(start),
//       endTimeBooking: formatLocal(end),
//     };
//   }

//   throw new Error("mode phải là 'week' hoặc 'month'");
// }

export function getBookingRange(start: Date, end: Date): BookingRange {
  const formatLocal = (d: Date) => d.toLocaleString("sv-SE", { hour12: false }).replace(" ", "T");

  return {
    startTimeBooking: formatLocal(start),
    endTimeBooking: formatLocal(end),
  };
  throw new Error("mode phải là 'week' hoặc 'month'");
}

// export function convertOldCarsToNew(oldCars: OldCar[]): Car[] {
//   return oldCars.map((oldCar) => {
//     // const date = oldCar.mode === "week" ? oldCar.startTime : oldCar.month;
//     // if (!date) throw new Error("OldCar thiếu thông tin ngày");

//     const { startTimeBooking, endTimeBooking } = getBookingRange(Date(oldCar.startTimeBooking), Date(oldCar.endTimeBooking));

//     return {
//       licensePlate: oldCar.licensePlate,
//       dateBooking: Date.now,
//       startTimeBooking,
//       endTimeBooking,
//       spotName: oldCar.spotName,
//       mode: oldCar.mode,
//     };
//   });
// }

export function convertOldCarsToNew(oldCars: OldCar[]): Car[] {
  const formatLocal = (d: Date) => d.toLocaleString("sv-SE", { hour12: false }).replace(" ", "T");

  return oldCars.map((oldCar) => {
    if (!oldCar.startTimeBooking || !oldCar.endTimeBooking) {
      throw new Error(`OldCar ${oldCar.licensePlate} thiếu thông tin ngày`);
    }

    const { startTimeBooking, endTimeBooking } = getBookingRange(
      new Date(oldCar.startTimeBooking),
      new Date(oldCar.endTimeBooking)
    );

    return {
      licensePlate: oldCar.licensePlate,
      dateBooking: formatLocal(new Date()), // ✅ chuyển sang string
      startTimeBooking,
      endTimeBooking,
      spotName: oldCar.spotName,
      mode: oldCar.mode,
    };
  });
}
