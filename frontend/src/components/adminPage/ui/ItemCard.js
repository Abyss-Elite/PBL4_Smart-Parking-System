import { Card, CardContent } from "@/components/ui/card";

export default function ItemCard({ title, content, classNameContent }) {
  return (
    <Card>
      <CardContent className="pt-4">
        <p className="text-sm text-gray-500">{title}</p>
        <p className={classNameContent}>{content}</p>
      </CardContent>
    </Card>
  );
}
