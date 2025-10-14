import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ExpiringTicketList({ expiringTickets}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Vé gửi xe sắp hết hạn</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead className="border-b">
            <tr className="text-left">
              <th className="py-2">Căn hộ</th>
              <th>Biển số</th>
              <th>Hạn vé</th>
            </tr>
          </thead>
          <tbody>
            {expiringTickets.map((ticket, index) => (
              <tr
                key={index}
                className={`border-b ${index === expiringTickets.length - 1 ? "last:border-none" : ""}`}
              >
                <td className="py-2">{ticket.apartment}</td>
                <td>{ticket.plate}</td>
                <td>{ticket.expiryDate}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
