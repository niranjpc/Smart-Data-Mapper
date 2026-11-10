// Minimal slice of the Office Scripts API used by ReconMonitor.ts (type-checking only).
declare namespace ExcelScript {
  interface Range { getValues(): (string | number | boolean)[][]; }
  interface Worksheet { getName(): string; getUsedRange(): Range | undefined; }
  interface Workbook { getWorksheets(): Worksheet[]; }
}
