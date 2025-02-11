unit cocktailunit2;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, TAGraph, TASeries, TATransformations, TATools,
  Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls, ComCtrls, LCLType,
  Globalvariables, Types, TACustomSeries, TADrawerSVG, TADrawUtils, TADrawerCanvas;

type

  { TForm2 }

  TForm2 = class(TForm)
    Bevel5: TBevel;

    Button1, Button2, Button3, Button4, Button5: TButton;

    Chart1: TChart;

    Chart1SensitiveOutLineSeries1, Chart1ResistantAOutLineSeries2, Chart1ResistantBOutLineSeries3,
    Chart1ResistantABOutLineSeries4, Chart1ResourceCOutLineSeries5, Chart1PhageAOutLineSeries6,
    Chart1PhageBOutLineSeries7, Chart1InfectedAOutLineSeries8, Chart1InfectedBOutLineSeries9,
    Chart1InfectedABOutLineSeries10, Chart1AResInfBOutLineSeries11, Chart1BResInfAOutLineSeries12,
    Chart1RefugeOutLineSeries13, Series: TLineSeries;

    ChartToolset1: TChartToolset;
    ChartToolset1DataPointClickTool1: TDataPointClickTool;
    ChartToolset1PanDragTool1: TPanDragTool;
    DataPointCrosshairTool1: TDataPointCrosshairTool;
    ChartToolset1ZoomMouseWheelTool1: TZoomMouseWheelTool;

    Label1: TLabel;

    Memo1: TMemo;
    SaveDialog1: TSaveDialog;
    SaveDialog2: TSaveDialog;
    SaveDialog3: TSaveDialog;

    LeftAxisTransformations: TChartAxisTransformations;
    LeftAxisAutoScaleTransform: TAutoScaleAxisTransform;
    RightAxisTransformations: TChartAxisTransformations;
    RightAxisAutoScaleTransform: TAutoScaleAxisTransform;
    StatusBar1: TStatusBar;

    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);

    procedure Button3Click(Sender: TObject);
    procedure Button4Click(Sender: TObject);
    procedure Button5Click(Sender: TObject);
    procedure Chart1MouseDown(Sender: TObject; Button: TMouseButton;
       Shift: TShiftState; X, Y: Integer);
    procedure ChartToolset1DataPointClickTool1PointClick(ATool: TChartTool;
       APoint: TPoint);

    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure Memo1Change(Sender: TObject);

    { private declarations }
  public
    { public declarations }
  end;

var
  Form2: TForm2;

implementation

{$R *.lfm}


{ TForm2 }


procedure TForm2.Chart1MouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  if (ssDouble in Shift) then
  begin
     DataPointCrosshairTool1.Enabled:= True;
     ChartToolset1DataPointClickTool1.Enabled := True;
     ChartToolset1ZoomMouseWheelTool1.Enabled := True;
     ChartToolset1PanDragTool1.Enabled := True;
     Statusbar1.SimpleText := 'Crosshair: Left mouse button at point ; Zoom: Mouse wheel ; Pan: Left mouse button down + drag ; Reset: Right mouse button';
   end
   else if (ssRight in Shift) then
   begin
     ChartToolset1ZoomMouseWheelTool1.Enabled := False;
     ChartToolset1PanDragTool1.Enabled := False;
     Chart1.ZoomFull;
     Statusbar1.SimpleText := 'Double click left mouse button in pane for zooming and panning. Click right button to return.';
   end;
end;

procedure TForm2.ChartToolset1DataPointClickTool1PointClick(ATool: TChartTool;
  APoint: TPoint);
var
  x, y: Single;
  LogPad, LogForm: String;
begin
  DefaultFormatSettings.DecimalSeparator:= '.';
  if DoLog then
  begin
    LogPad := 'log10 of ';
    LogForm := ' = %2f';
  end
  else
  begin
    LogPad := '';
    LogForm := ' = %8.3e';
  end;
  with ATool as TDatapointClickTool do
    if (Series is TLineSeries) then
      with TLineSeries(Series) do begin
        x := GetXValue(PointIndex);
        y := GetYValue(PointIndex);
        if DoLog and (y = -16) then y := 0;
        case Title of
        'Resource' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Concentration/ml' + ' = %f', [Title, x, y]);
        'Uninfected' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Uninfected cells, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'A Resistant' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage A resistant cells, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'B Resistant' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage B resistant cells, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'AB Resistant' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage A and B resistant cells, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'Phage A' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage A titre, ' + LogPad + 'PFU/ml' + LogForm, [Title, x, y]);
        'Phage B' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage B titre, ' + LogPad + 'PFU/ml' + LogForm, [Title, x, y]);
        'Infected by A' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Cells infected by A, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'Infected by B' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Cells infected by B, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'Infected by AB' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Cells infected by AB, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'A resistant infected by B' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage A resistant cells infected by B, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'B resistant infected by A' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Phage B resistant cells infected by A, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        'Refuge cells' : Statusbar1.SimpleText := Format('%s: Time = %f ; ' + 'Refuge cells: Uninfected + Resistant to A, B or AB, ' + LogPad + 'CFU/ml' + LogForm, [Title, x, y]);
        end;
        end
        else Statusbar1.SimpleText := '';
end;

procedure TForm2.Button1Click(Sender: TObject);
begin
  Close;
end;

procedure TForm2.Button2Click(Sender: TObject);
var
  DataStrings: TStringList;
begin
  DataStrings := TStringList.Create;
  SaveDialog1.DefaultExt := 'ctl';
  SaveDialog1.Options := [ofOverwritePrompt];
  if SaveDialog1.Execute then
    try
    DataStrings.Add('= Cocktail input file: ' + SaveDialog1.Filename + '. Add label (one line):');
    DataStrings.Add(' ');
    DataStrings.AddStrings(Memo1.Lines);
    DataStrings.Add(' ');
    DataStrings.AddStrings(OutputParam);
    DataStrings.Add(' ');
    DataStrings.Add('Comments can be added below:');
    DataStrings.SaveToFile(SaveDialog1.Filename);
    except
    on E: Exception do ShowMessage('File ' + SaveDialog1.Filename + ' could not be written. Error: ' + E.Message);
    end;
    DataStrings.Free;
end;

procedure TForm2.Button3Click(Sender: TObject);
begin
  SaveDialog2.Options := [ofOverwritePrompt];
  if SaveDialog2.Execute then
    try
      Chart1.SaveToFile(TPortableNetworkGraphic, SaveDialog2.Filename);
    except
      on E: Exception do ShowMessage('File ' + SaveDialog2.Filename + ' could not be written. Error: ' + E.Message);
    end;
end;

procedure TForm2.Button4Click(Sender: TObject);
var
  fs: TFileStream;
  id: IChartDrawer;
begin
  SaveDialog3.Options := [ofOverwritePrompt];
  if SaveDialog3.Execute then
  begin
  try
    fs := TFileStream.Create(SaveDialog3.Filename, fmCreate);
    id := TSVGDrawer.Create(fs, true);
    with Chart1 do
      Draw(id, Rect(0, 0, Width, Height));
  except
     on E: Exception do ShowMessage('File ' + SaveDialog3.Filename + ' could not be written. Error: ' + E.Message);
  end;
     fs.Free;
  end;
end;

procedure TForm2.Button5Click(Sender: TObject);
begin
  Chart1.CopyToClipboardBitmap;
end;

procedure TForm2.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  CloseAction := caFree;
end;

procedure TForm2.FormCreate(Sender: TObject);
begin
  Statusbar1.SimpleText := 'Double click left mouse button in pane for detailed view. Click right button to return.';
end;

procedure TForm2.Memo1Change(Sender: TObject);
begin
  Memo1.Lines.Clear;
end;

end.

