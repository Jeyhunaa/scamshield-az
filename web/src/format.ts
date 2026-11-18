const amountFmt = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const money = (value: number): string => amountFmt.format(value);

export const compactIban = (iban: string): string => iban.replace(/\s+/g, "").toUpperCase();

export const prettyIban = (iban: string): string => compactIban(iban).replace(/(.{4})(?=.)/g, "$1 ");

export const maskedIban = (iban: string): string => {
  const s = compactIban(iban);
  return s.length < 8 ? s : `${s.slice(0, 4)} •••• ${s.slice(-4)}`;
};

export const isValidAzIban = (iban: string): boolean => /^AZ\d{2}[A-Z0-9]{24}$/.test(compactIban(iban));

export const initials = (name: string): string =>
  name
    .split(/\s+/)
    .filter((p) => /^[A-Za-zÀ-ž]/.test(p))
    .slice(0, 2)
    .map((p) => p[0].toUpperCase())
    .join("") || "?";

export const clockTime = (d = new Date()): string =>
  d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });

export const longDate = (d = new Date()): string =>
  d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) + ", " + clockTime(d);

export const reference = (): string => {
  const d = new Date();
  const ymd = `${String(d.getFullYear()).slice(2)}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  return `DB${ymd}-${Math.floor(100000 + Math.random() * 900000)}`;
};

export const greeting = (d = new Date()): string => {
  const h = d.getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
};
