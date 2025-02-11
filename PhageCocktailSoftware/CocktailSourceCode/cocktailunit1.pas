unit cocktailunit1;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls, Math,
  ExtCtrls, Spin, ComCtrls, LCLIntf, Globalvariables, cocktailunit2;
  function GetLineNumbers(InputNumber: Single): Single;
  function MutStochastic(Cellnumber, Mutfreq: Real): Real;
  function GetInputNumber(NumberString: string; RangeFrom, RangeTo, DefNumber: Single; IsReal: Boolean): Single;
  procedure ReadCTLFile;
  procedure CellsToRefuge;
  procedure CellsFromRefuge;

type

  { TForm1 }

  TForm1 = class(TForm)
    Bevel1, Bevel2, Bevel3, Bevel4, Bevel5, Bevel6: TBevel;

    Button1, Button2, Button3, Button4: TButton;

    OpenDialog1: TOpenDialog;

    rgTimeStep, rgMutationMod, rgInfectionMod, rgRefugeCells, rgAdsorptionMod: TRadioGroup;

    SensitiveOut, InfectedAOut, ResistantAOut, ResistantBOut, ResistantABOut, ResourceCOut,
    PhageAOut, PhageBOut, LogScale, AResInfbyBOut, BResInfbyAOut, RefugeOut,
    InfectedABOut, InfectedBOut, RoundOff: TCheckBox;

    Edit1, Edit2, Edit3, Edit4, Edit5, Edit6, Edit7, Edit8, Edit9, Edit10, Edit11: TEdit;

    FloatSpinEdit1, FloatSpinEdit2, FloatSpinEdit3, FloatSpinEdit4, FloatSpinEdit5,
    FloatSpinEdit6, FloatSpinEdit7, FloatSpinEdit8, FloatSpinEdit9, FloatSpinEdit10, FloatSpinEdit11: TFloatSpinEdit;

    Label1, Label2, Label3, Label4, Label5, Label6, Label7, Label8, Label9,
    Label10, Label11, Label12, Label13, Label14, Label15, Label16, Label17,
    Label18, Label19, Label20, Label21, Label22, Label23, Label24, Label25,
    Label26, Label27, Label28, Label29, Label30, Label31, Label32, Label33,
    Label34, Label36, Label37: TLabel;

    SpinEdit1, SpinEdit2, SpinEdit3, SpinEdit4, SpinEdit5,
    SpinEdit7, SpinEdit8, SpinEdit9, SpinEdit10, SpinEdit11,
    SpinEdit13, SpinEdit14, SpinEdit15, SpinEdit16: TSpinEdit;

    StatusBar1: TStatusBar;

    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure Button4Click(Sender: TObject);

    procedure Edit1Change(Sender: TObject);
    procedure Edit2Change(Sender: TObject);
    procedure Edit3Change(Sender: TObject);
    procedure Edit4Change(Sender: TObject);
    procedure Edit5Change(Sender: TObject);
    procedure Edit6Change(Sender: TObject);
    procedure Edit7Change(Sender: TObject);
    procedure Edit8Change(Sender: TObject);
    procedure Edit9Change(Sender: TObject);
    procedure Edit10Change(Sender: TObject);
    procedure Edit11Change(Sender: TObject);

    procedure Edit1EditingDone(Sender: TObject);
    procedure Edit2EditingDone(Sender: TObject);
    procedure Edit3EditingDone(Sender: TObject);
    procedure Edit4EditingDone(Sender: TObject);
    procedure Edit5EditingDone(Sender: TObject);
    procedure Edit6EditingDone(Sender: TObject);
    procedure Edit7EditingDone(Sender: TObject);
    procedure Edit8EditingDone(Sender: TObject);
    procedure Edit9EditingDone(Sender: TObject);
    procedure Edit10EditingDone(Sender: TObject);
    procedure Edit11EditingDone(Sender: TObject);

    procedure FloatSpinEdit1Change(Sender: TObject);
    procedure FloatSpinEdit2Change(Sender: TObject);
    procedure FloatSpinEdit3Change(Sender: TObject);
    procedure FloatSpinEdit4Change(Sender: TObject);
    procedure FloatSpinEdit5Change(Sender: TObject);
    procedure FloatSpinEdit6Change(Sender: TObject);
    procedure FloatSpinEdit7Change(Sender: TObject);
    procedure FloatSpinEdit8Change(Sender: TObject);
    procedure FloatSpinEdit9Change(Sender: TObject);
    procedure FloatSpinEdit10Change(Sender: TObject);
    procedure FloatSpinEdit11Change(Sender: TObject);

    procedure rgInfectionModClick(Sender: TObject);
    procedure rgMutationModClick(Sender: TObject);
    procedure rgAdsorptionModClick(Sender: TObject);
    procedure rgRefugeCellsClick(Sender: TObject);
    procedure rgTimeStepClick(Sender: TObject);

    procedure SpinEdit10Change(Sender: TObject);
    procedure SpinEdit11Change(Sender: TObject);

    procedure SpinEdit13Change(Sender: TObject);
    procedure SpinEdit14Change(Sender: TObject);
    procedure SpinEdit15Change(Sender: TObject);
    procedure SpinEdit16Change(Sender: TObject);
    procedure SpinEdit1Change(Sender: TObject);
    procedure SpinEdit2Change(Sender: TObject);
    procedure SpinEdit3Change(Sender: TObject);
    procedure SpinEdit4Change(Sender: TObject);
    procedure SpinEdit5Change(Sender: TObject);

    procedure SpinEdit7Change(Sender: TObject);
    procedure SpinEdit8Change(Sender: TObject);
    procedure SpinEdit9Change(Sender: TObject);

    procedure FormCreate(Sender: TObject);
    procedure ShowMe(Sender: TObject);

  private

    { private declarations }
  public { public declarations }

  end;

var
  Form1: TForm1;
  I, IOutput: Integer;
  DataStrings: TStringList;
  filename: string;

implementation

{$R *.lfm}

{ Assigning and checking changed values }

procedure TForm1.FloatSpinEdit1Change(Sender: TObject);
begin
  MaxGrowthSens := FloatSpinEdit1.Value;
  StatusBar1.SimpleText := 'Bacterial max growth rate has changed';
end;

procedure TForm1.FloatSpinEdit2Change(Sender: TObject);
begin
  MonodK := FloatSpinEdit2.Value;
  StatusBar1.SimpleText := 'Monod constant has changed.';
end;

procedure TForm1.FloatSpinEdit3Change(Sender: TObject);
begin
  DecayRateBact := FloatSpinEdit3.Value;
  StatusBar1.SimpleText := 'Bacterial decay rate has changed';
end;

procedure TForm1.FloatSpinEdit4Change(Sender: TObject);
begin
  FlowRate := FloatSpinEdit4.Value;
  StatusBar1.SimpleText := 'Resource flow rate has changed';
end;

procedure TForm1.FloatSpinEdit5Change(Sender: TObject);
begin
  DecayRatePhageA := FloatSpinEdit5.Value;
  StatusBar1.SimpleText := 'Phage A decay rate has changed';
end;

procedure TForm1.FloatSpinEdit6Change(Sender: TObject);
begin
  DecayRatePhageB := FloatSpinEdit6.Value;
  StatusBar1.SimpleText := 'Phage B decay rate has changed';
end;

procedure TForm1.FloatSpinEdit7Change(Sender: TObject);
begin
  MaxGrowthResA := FloatSpinEdit7.Value;
  StatusBar1.SimpleText := 'A-resistant bacterial max growth rate has changed';
end;

procedure TForm1.FloatSpinEdit8Change(Sender: TObject);
begin
  MaxGrowthResB := FloatSpinEdit8.Value;
  StatusBar1.SimpleText := 'B-resistant bacterial max growth rate has changed';
end;

procedure TForm1.FloatSpinEdit9Change(Sender: TObject);
begin
  MaxGrowthResAB := FloatSpinEdit9.Value;
  StatusBar1.SimpleText := 'AB-resistant bacterial max growth rate has changed';
end;

procedure TForm1.rgInfectionModClick(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Primary adsorption model has changed';
end;

procedure TForm1.rgAdsorptionModClick(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Secondary adsorption mode has changed';
end;

procedure TForm1.rgMutationModClick(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Phage resistance mutation model has changed';
end;

procedure TForm1.rgRefugeCellsClick(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Refuge cells type has changed';
end;

procedure TForm1.rgTimeStepClick(Sender: TObject);
begin
   StatusBar1.SimpleText := 'Time step size has changed';
end;

procedure TForm1.ShowMe(Sender: TObject);
begin
  ShowMessage('                                 Cocktail 2.3.4 beta' + sLineBreak +
  'The Cocktail program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit:' + sLineBreak +
  'https://creativecommons.org/licenses/by-nc-sa/4.0/' + sLineBreak +
  'Developed by Anders S. Nilsson, Department of Molecular Biosciences, The Wenner-Gren Institute, Stockholm University, Sweden, anders.s.nilsson@su.se');
end;

procedure TForm1.SpinEdit1Change(Sender: TObject);
begin
  ResourceStartC := SpinEdit1.Value;
  StatusBar1.SimpleText := 'Resource start concentration has changed';
end;

procedure TForm1.SpinEdit2Change(Sender: TObject);
begin
  ResevoirC := SpinEdit2.Value;
  StatusBar1.SimpleText := 'Resource inflow concentration has changed';
end;

procedure TForm1.SpinEdit3Change(Sender: TObject);
begin
  RunningTimeHour := SpinEdit3.Value;
  StatusBar1.SimpleText := 'Running time in hours has changed';
end;

procedure TForm1.SpinEdit4Change(Sender: TObject);
begin
  LatencyA := SpinEdit4.Value;
  StatusBar1.SimpleText := 'Phage A latency time has changed';
end;

procedure TForm1.SpinEdit5Change(Sender: TObject);
begin
  BurstA := SpinEdit5.Value;
  StatusBar1.SimpleText := 'Phage A burst size has changed';
end;

procedure TForm1.SpinEdit7Change(Sender: TObject);
begin
  AddTimeOneA := SpinEdit7.Value;
  StatusBar1.SimpleText := 'First addition time of phage A has changed';
end;

procedure TForm1.SpinEdit8Change(Sender: TObject);
begin
  AddTimeTwoA := SpinEdit8.Value;
  StatusBar1.SimpleText := 'Second addition time of phage A has changed';
end;

procedure TForm1.SpinEdit9Change(Sender: TObject);
begin
  AddTimeThreeA:= SpinEdit9.Value;
  StatusBar1.SimpleText := 'Third addition time of phage A has changed';
end;
procedure TForm1.SpinEdit10Change(Sender: TObject);
begin
  LatencyB := SpinEdit10.Value;
  StatusBar1.SimpleText := 'Phage B latency time has changed';
end;

procedure TForm1.SpinEdit11Change(Sender: TObject);
begin
  BurstB := SpinEdit11.Value;
  StatusBar1.SimpleText := 'Phage B burst size has changed';
end;

procedure TForm1.SpinEdit13Change(Sender: TObject);
begin
  AddTimeOneB := SpinEdit13.Value;
  StatusBar1.SimpleText := 'First addition time of phage B has changed';
end;

procedure TForm1.SpinEdit14Change(Sender: TObject);
begin
  AddTimeTwoB := SpinEdit14.Value;
  StatusBar1.SimpleText := 'Second addition time of phage B has changed';
end;

procedure TForm1.SpinEdit15Change(Sender: TObject);
begin
  AddTimeThreeB:= SpinEdit15.Value;
  StatusBar1.SimpleText := 'Third addition time of phage B has changed';
end;

procedure TForm1.SpinEdit16Change(Sender: TObject);
begin
  RunningTimeMinute := SpinEdit16.Value;
  StatusBar1.SimpleText := 'Running time in minutes has changed';
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  DefaultFormatSettings.DecimalSeparator:= '.';
  if ParamStr(1) <> '' then
  begin
    if FileExists(ParamStr(1)) then
      begin
        filename := ParamStr(1);
        ReadCTLFile;
      end else ShowMessage('File not found');
  end;
end;

function GetInputNumber(NumberString: string; RangeFrom, RangeTo, DefNumber: Single; IsReal: Boolean): Single;
var
RealNumber: Single;
begin
  DefaultFormatSettings.DecimalSeparator:= '.';
  if Pos(',',NumberString) <> 0 then NumberString[Pos(',',NumberString)] := '.';
  if TryStrToFloat(NumberString, RealNumber) then
    begin
      if (RealNumber < RangeFrom) or (RealNumber > RangeTo) then
        begin
          ShowMessage('The number is out of range. Check allowed range in the hint.');
          GetInputNumber := DefNumber;
        end
        else GetInputNumber := RealNumber;
    end
    else
      begin
        if (NumberString = '') or (NumberString = ' ') then
          begin
            RealNumber := 0;
            if (RealNumber < RangeFrom) then
              begin
                ShowMessage('The number is out of range. Check allowed range in the hint.');
                GetInputNumber := DefNumber;
              end
              else GetInputNumber := 0;
          end
          else
          begin
            if IsReal then ShowMessage('Incorrect value. Enter a real number or a number'
            + sLineBreak + 'in scientific notation (e.g. 1.0E-6)')
            else
            ShowMessage('Incorrect value. Enter an integer or a number'
            + sLineBreak + 'in scientific notation (e.g. 1.0E+8)');
            GetInputNumber := DefNumber;
          end;
      end;
end;

procedure TForm1.Edit1EditingDone(Sender: TObject);
begin
  ConversEff := GetInputNumber(Edit1.Text,1.00E-8,1.00E-4,2.00E-6,true);
  Edit1.Text := FloatToStrF(ConversEff,ffExponent,3,1);
end;

procedure TForm1.Edit1Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Conversion efficiency has changed';
end;

procedure TForm1.Edit2EditingDone(Sender: TObject);
begin
  ResistRateA := GetInputNumber(Edit2.Text,0,1.00E-2,1.00E-7,true);
  Edit2.Text := FloatToStrF(ResistRateA,ffExponent,3,1);
end;

procedure TForm1.Edit2Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Resistance rate to phage A has changed';
end;

procedure TForm1.Edit3EditingDone(Sender: TObject);
begin
  ResistRateB := GetInputNumber(Edit3.Text,0,1.00E-2,1.00E-7,true);
  Edit3.Text := FloatToStrF(ResistRateB,ffExponent,3,1);
end;

procedure TForm1.Edit3Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Resistance rate to phage B has changed';
end;

procedure TForm1.Edit4EditingDone(Sender: TObject);
begin
  ResistantToA := GetInputNumber(Edit4.Text,0,1.00E-3,1.00E-7,true);
  Edit4.Text := FloatToStrF(ResistantToA,ffExponent,3,1);
end;

procedure TForm1.Edit4Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Start bacteria resistant to phage A has changed';
end;

procedure TForm1.Edit5EditingDone(Sender: TObject);
begin
  ResistantToB := GetInputNumber(Edit5.Text,0,1.00E-3,1.00E-7,true);
  Edit5.Text := FloatToStrF(ResistantToB,ffExponent,3,1);
end;

procedure TForm1.Edit5Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Start bacteria resistant to phage B has changed';
end;

procedure TForm1.Edit6EditingDone(Sender: TObject);
begin
  PhageTitreA := GetInputNumber(Edit6.Text,0,1.00E+13,1.00E+8,false);
  Edit6.Text := FloatToStrF(PhageTitreA,ffExponent,3,2);
end;

procedure TForm1.Edit6Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Phage A starting titre has changed';
end;

procedure TForm1.Edit7EditingDone(Sender: TObject);
begin
  PhageTitreB := GetInputNumber(Edit7.Text,0,1.00E+13,1.00E+8,false);
  Edit7.Text := FloatToStrF(PhageTitreB,ffExponent,3,2);
end;

procedure TForm1.Edit7Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Phage B starting titre has changed';
end;

procedure TForm1.Edit8EditingDone(Sender: TObject);
begin
  Sensitive := GetInputNumber(Edit8.Text,10,1.00E+12,1.00E+8,false);
  Edit8.Text := FloatToStrF(Sensitive,ffExponent,3,2);
end;

procedure TForm1.Edit8Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Bacteria starting titre has changed';
end;

procedure TForm1.Edit9EditingDone(Sender: TObject);
begin
  AdsorbRateA := GetInputNumber(Edit9.Text,1.00E-14,1.00E-7,1.00E-10,true);
  Edit9.Text := FloatToStrF(AdsorbRateA,ffExponent,3,2);
end;

procedure TForm1.Edit9Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Adsorption rate of phage A has changed';
end;

procedure TForm1.Edit10EditingDone(Sender: TObject);
begin
  AdsorbRateB := GetInputNumber(Edit10.Text,1.00E-14,1.00E-7,1.00E-10,true);
  Edit10.Text := FloatToStrF(AdsorbRateB,ffExponent,3,2);
end;

procedure TForm1.Edit10Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Adsorption rate of phage B has changed';
end;

procedure TForm1.Edit11EditingDone(Sender: TObject);
begin
  ResistantToAB := GetInputNumber(Edit11.Text,0,1.00E-6,1.00E-14,true);
  Edit11.Text := FloatToStrF(ResistantToAB,ffExponent,3,2);
end;

procedure TForm1.Edit11Change(Sender: TObject);
begin
  StatusBar1.SimpleText := 'Start bacteria resistant to phage A and B has changed';
end;

procedure TForm1.FloatSpinEdit10Change(Sender: TObject);
begin
  RefugeRateIn := FloatSpinEdit10.Value;
  StatusBar1.SimpleText := 'Rate of bacteria into a refuge population has changed';
end;

procedure TForm1.FloatSpinEdit11Change(Sender: TObject);
begin
  RefugeRateOut := FloatSpinEdit11.Value;
  StatusBar1.SimpleText := 'Rate of bacteria from a refuge population has changed';
end;

function GetLineNumbers(InputNumber: Single): Single;  {Get line data and check for values < 0 (for Log10 <1) and round off values}
begin
  if (InputNumber < 1) and (Form1.RoundOff.Checked = True) then InputNumber := 0;
  if InputNumber <= 0 then GetLineNumbers := 0 else GetLineNumbers := InputNumber;
  if Form1.LogScale.Checked = True then
  begin
    if (InputNumber < 0.0000000000000001) then GetLineNumbers := Log10(0.0000000000000001) else GetLineNumbers := Log10(InputNumber);
  end;
end;

{Mutations are calculated randomly sampling from a Poisson distribution
up to Lambda <=10 or from normal distribution approximation from lambda > 10}
function MutStochastic(Cellnumber, Mutfreq: Real): Real;
var
  RandNumber, Lambda, AccPoisson: Real;
  Ix: integer;
  IFac: Longint;
 begin
   Lambda := Cellnumber*Mutfreq;
   if Lambda <= 10 then
   begin
     AccPoisson := 0;
     IFac := 1;
     Ix := 0;
     RandNumber := Random;
     repeat
       if Ix >= 1 then IFac := Ix*IFac;
       AccPoisson := AccPoisson + ((Lambda**Ix)*(Exp(-Lambda))/IFac);
       Inc(Ix);
     until (AccPoisson > RandNumber);
     MutStochastic := Ix - 1;
   end
   else MutStochastic := randg(Lambda,sqrt(Lambda));
end;

procedure CellsToRefuge;
begin
  if AllSusceptible > 0 then
    begin
      AckRefugeSensitive[IRefugePopNr] := AckRefugeSensitive[IRefugePopNr] + (Sensitive * RefugeRateIn);
      Sensitive := Sensitive - (Sensitive * RefugeRateIn);
      AckRefugeResistantToA[IRefugePopNr] := AckRefugeResistantToA[IRefugePopNr] + (ResistantToA * RefugeRateIn);
      ResistantToA := ResistantToA - (ResistantToA * RefugeRateIn);
      AckRefugeResistantToB[IRefugePopNr] := AckRefugeResistantToB[IRefugePopNr] + (ResistantToB * RefugeRateIn);
      ResistantToB := ResistantToB - (ResistantToB * RefugeRateIn);
      AckRefugeResistantToAB[IRefugePopNr] := AckRefugeResistantToAB[IRefugePopNr] + (ResistantToAB * RefugeRateIn);
      ResistantToAB := ResistantToAB - (ResistantToAB * RefugeRateIn);
      AckRefugeCells[IRefugePopNr] := AckRefugeSensitive[IRefugePopNr] +  AckRefugeResistantToA[IRefugePopNr] + AckRefugeResistantToB[IRefugePopNr] + AckRefugeResistantToAB[IRefugePopNr];
    end;
end;

procedure CellsFromRefuge;
begin
  if AckRefugeCells[IRefugePopNr] > 0 then
    begin
      Sensitive := Sensitive + (AckRefugeSensitive[IRefugePopNr] * TotalRefugeRateOut);
      AckRefugeSensitive[IRefugePopNr] := AckRefugeSensitive[IRefugePopNr] - (AckRefugeSensitive[IRefugePopNr] * TotalRefugeRateOut);
      ResistantToA := ResistantToA + (AckRefugeResistantToA[IRefugePopNr] * TotalRefugeRateOut);
      AckRefugeResistantToA[IRefugePopNr] := AckRefugeResistantToA[IRefugePopNr] - (AckRefugeResistantToA[IRefugePopNr] * TotalRefugeRateOut);
      ResistantToB := ResistantToB + (AckRefugeResistantToB[IRefugePopNr] * TotalRefugeRateOut);
      AckRefugeResistantToB[IRefugePopNr] := AckRefugeResistantToB[IRefugePopNr] - (AckRefugeResistantToB[IRefugePopNr] * TotalRefugeRateOut);
      ResistantToAB := ResistantToAB + (AckRefugeResistantToAB[IRefugePopNr] * TotalRefugeRateOut);
      AckRefugeResistantToAB[IRefugePopNr] := AckRefugeResistantToAB[IRefugePopNr] - (AckRefugeResistantToAB[IRefugePopNr] * TotalRefugeRateOut);
      AckRefugeCells[IRefugePopNr] := AckRefugeSensitive[IRefugePopNr] +  AckRefugeResistantToA[IRefugePopNr] + AckRefugeResistantToB[IRefugePopNr] + AckRefugeResistantToAB[IRefugePopNr];
    end;
end;

{ Build input form TForm1 and prepare output form NewForm}

procedure TForm1.Button1Click(Sender: TObject);
var
  NewForm: TForm2;
  TimeStep: Integer;
begin
  ClearVariables;

  {Set starting values}
  MaxGrowthSens := FloatSpinEdit1.Value;
  MonodK :=  FloatSpinEdit2.Value;
  DecayRateBact := FloatSpinEdit3.Value;
  FlowRate := FloatSpinEdit4.Value;
  DecayRatePhageA := FloatSpinEdit5.Value;
  DecayRatePhageB := FloatSpinEdit6.Value;
  MaxGrowthResA := FloatSpinEdit7.Value;
  MaxGrowthResB := FloatSpinEdit8.Value;
  MaxGrowthResAB := FloatSpinEdit9.Value;
  RefugeRateIn := FloatSpinEdit10.Value;
  RefugeRateOut := FloatSpinEdit11.Value;

  AdsorbRateA := GetInputNumber(Edit9.Text,1.00E-14,1.00E-7,1.00E-10,true);
  AdsorbRateB := GetInputNumber(Edit10.Text,1.00E-14,1.00E-7,1.00E-10,true);
  LatencyA := SpinEdit4.Value;
  LatencyB := SpinEdit10.Value;
  BurstA := SpinEdit5.Value;
  BurstB := SpinEdit11.Value;

  PhageTitreA := GetInputNumber(Edit6.Text,0,1.00E+13,1.00E+8,false);
  PhageTitreB := GetInputNumber(Edit7.Text,0,1.00E+13,1.00E+8,false);
  AddTimeOneA := SpinEdit7.Value;
  AddTimeTwoA := SpinEdit8.Value;
  AddTimeThreeA:= SpinEdit9.Value;
  AddTimeOneB := SpinEdit13.Value;
  AddTimeTwoB := SpinEdit14.Value;
  AddTimeThreeB:= SpinEdit15.Value;

  ResourceStartC := SpinEdit1.Value;
  ResevoirC := SpinEdit2.Value;
  RunningTimeHour := SpinEdit3.Value;
  RunningTimeMinute := SpinEdit16.Value;

  ConversEff := GetInputNumber(Edit1.Text,1.00E-8,1.00E-4,2.00E-6,true);
  ResistRateA := GetInputNumber(Edit2.Text,0,1.00E-2,1.00E-7,true);
  ResistRateB := GetInputNumber(Edit3.Text,0,1.00E-2,1.00E-7,true);
  ResistantToA := GetInputNumber(Edit4.Text,0,1.00E-3,1.00E-7,true);
  ResistantToB := GetInputNumber(Edit5.Text,0,1.00E-3,1.00E-7,true);
  ResistantToAB := GetInputNumber(Edit11.Text,0,1.00E-6,1.00E-14,true);
  Sensitive :=  GetInputNumber(Edit8.Text,10,1.00E+12,1.00E+8,false);

  {End set starting values}

  {Build Output form}
  NewForm := TForm2.Create(Application);
  Inc(IOutput);
  With NewForm do
  begin
    Caption := 'Cocktail output: ' + IntToStr(IOutput);
    With Memo1.Lines do
    begin
      Clear;
      Add('Bacteria');
      Add('Max growth rate uninfected = ' + FloatToStr(FloatSpinEdit1.Value));
      Add('Monod constant = ' + FloatToStr(FloatSpinEdit2.Value));
      Add('Conversion efficiency = ' + Edit1.Text);
      Add('Bacterial decay rate = ' + FloatToStr(FloatSpinEdit3.Value));
      Add('A resistance mutation rate = ' + Edit2.Text);
      Add('B resistance mutation rate = ' + Edit3.Text);
      Add('A resistant frequency = ' + Edit4.Text);
      Add('B resistant frequency = ' + Edit5.Text);
      Add('AB resistant frequency = ' + Edit11.Text);
      Add('Max growth rate A resistant = ' + FloatToStr(FloatSpinEdit7.Value));
      Add('Max growth rate B resistant = ' + FloatToStr(FloatSpinEdit8.Value));
      Add('Max growth rate AB resistant = ' + FloatToStr(FloatSpinEdit9.Value));
      Add('Start titre = ' + Edit8.Text);
      Add(' ');
      Add('Resources');
      Add( 'Start concentration = ' + IntToStr(SpinEdit1.Value));
      Add('Resevoir concentration = ' + IntToStr(SpinEdit2.Value));
      Add('Flow rate = ' + FloatToStr(FloatSpinEdit4.Value));
      Add(' ');
      Add('Phages');
      Add('Adsorption rates A = '+ Edit9.Text + ', B = ' + Edit10.Text);
      Add('Latency times A = '+ IntToStr(SpinEdit4.Value) + ', B = ' + IntToStr(SpinEdit10.Value));
      Add('Burst sizes A = '+ IntToStr(SpinEdit5.Value) + ', B = ' + IntToStr(SpinEdit11.Value));
      Add('Decay rates A = '+ FloatToStr(FloatSpinEdit5.Value) + ', B = ' + FloatToStr(FloatSpinEdit6.Value));
      Add('Added titres A = '+ Edit6.Text + ', B = ' + Edit7.Text);
      Add('A additions (minutes) = '+ IntToStr(SpinEdit7.Value) + ', ' + IntToStr(SpinEdit8.Value) +', ' + IntToStr(SpinEdit9.Value));
      Add('B additions (minutes) = '+ IntToStr(SpinEdit13.Value) + ', ' + IntToStr(SpinEdit14.Value) +', ' + IntToStr(SpinEdit15.Value));
      Add(' ');
      Add('Models');
      Add('Primary adsorption: ' + rgInfectionMod.Items[rgInfectionMod.ItemIndex]);
      Add('Secondary adsorption: ' + rgAdsorptionMod.Items[rgAdsorptionMod.ItemIndex]);
      Add('Mutation: ' + rgMutationMod.Items[rgMutationMod.ItemIndex]);
      Add('Refuge cells: ' + rgRefugeCells.Items[rgRefugeCells.ItemIndex]);
      Add('Rates: In = ' + FloatToStr(FloatSpinEdit10.Value) + ', Out = ' + FloatToStr(FloatSpinEdit11.Value));
      Add(' ');
      Add('Time step size: ' + rgTimeStep.Items[rgTimeStep.ItemIndex]);
      Add('Running time (hours and minutes): ' + IntToStr(SpinEdit3.Value) + ', ' + IntToStr(SpinEdit16.Value));
    end;
  end;

  ClearOldSet;

  {Set time dependent starting values}

case rgTimeStep.ItemIndex of
       0 : begin
             HourDivisor := 720; {5 sec}
             MinuteDivisor := 12;
           end;
       1 : begin
             HourDivisor := 240; {15 sec}
             MinuteDivisor := 4;
           end;
       2 : begin
             HourDivisor := 120; {30 sec}
             MinuteDivisor := 2;
           end;
       3 : begin
             HourDivisor := 60; {1 min}
             MinuteDivisor := 1;
           end;
end; {case TimeStep}

  AdsorbRateA := AdsorbRateA / MinuteDivisor;
  AdsorbRateB := AdsorbRateB / MinuteDivisor;
  DecayRatePhageA := DecayRatePhageA/ HourDivisor;
  DecayRatePhageB := DecayRatePhageB/ HourDivisor;
  LatencyA := LatencyA * MinuteDivisor;
  LatencyB := LatencyB * MinuteDivisor;
  AddTimeOneA := AddTimeOneA * MinuteDivisor;
  AddTimeOneB := AddTimeOneB * MinuteDivisor;
  AddTimeTwoA := AddTimeTwoA * MinuteDivisor;
  AddTimeTwoB := AddTimeTwoB * MinuteDivisor;
  AddTimeThreeA := AddTimeThreeA * MinuteDivisor;
  AddTimeThreeB := AddTimeThreeB * MinuteDivisor;
  MaxGrowthSens := MaxGrowthSens/ HourDivisor;
  MaxGrowthResA := MaxGrowthResA/ HourDivisor;
  MaxGrowthResB := MaxGrowthResB/ HourDivisor;
  MaxGrowthResAB := MaxGrowthResAB/ HourDivisor;
  DecayRateBact := DecayRateBact/ HourDivisor;
  RefugeRateIn := RefugeRateIn/ MinuteDivisor;
  RefugeRateOut := RefugeRateOut/ MinuteDivisor;
  {Resources}
  FlowRate := FlowRate/ HourDivisor;
  MonodFunc := ResourceStartC/(ResourceStartC+MonodK);
  ResourceC := ResourceStartC;
  {Set dynamic dimension of arrays}
  SetLength(ConveyorA, LatencyA);
  SetLength(ConveyorB, LatencyB);
  SetLength(ITimeToLysisA, LatencyA);
  SetLength(ITimeToLysisB, LatencyB);
  if LatencyA <= LatencyB then
    begin
      LatencyAB := LatencyA;
      SetLength(ConveyorAB, LatencyA);
    end
    else
    begin
      LatencyAB := LatencyB;
      SetLength(ConveyorAB, LatencyB);
    end;
  SetLength(ConveyorBResA, LatencyA);
  SetLength(ConveyorAResB, LatencyB);
  SetLength(ConveyorAonBA, LatencyA);
  SetLength(ConveyorAonBB, LatencyB);
  SetLength(ConveyorBonAA, LatencyA);
  SetLength(ConveyorBonAB, LatencyB);
  SetLength(ConveyorBoundA, LatencyA);
  SetLength(ConveyorBoundB, LatencyB);
  SetLength(ConveyorFreqInfectedB, LatencyB);
  for ICount := 0 to LatencyA-1 do
        begin
          ConveyorA[ICount] := 0;
          ITimeToLysisA[ICount] := 0;
          if LatencyA <= LatencyB then ConveyorAB[ICount] := 0;
          ConveyorBResA[ICount] := 0;
          ConveyorAonBA[ICount] := 0;
          ConveyorBonAA[ICount] := 0;
          ConveyorBoundA[ICount] := 0;
        end;
  for ICount := 0 to LatencyB-1 do
        begin
          ConveyorB[ICount] := 0;
          ITimeToLysisB[ICount] := 0;
          if LatencyA > LatencyB then ConveyorAB[ICount] := 0;
          ConveyorAResB[ICount] := 0;
          ConveyorAonBB[ICount] := 0;
          ConveyorBonAB[ICount] := 0;
          ConveyorBoundB[ICount] := 0;
        end;

  SetLength(AckRefugeSensitive, 1);
  SetLength(AckRefugeResistantToA, 1);
  SetLength(AckRefugeResistantToB, 1);
  SetLength(AckRefugeResistantToAB, 1);
  SetLength(AckRefugeCells, 1);
  AckRefugeSensitive[0] := 0;
  AckRefugeResistantToA[0] := 0;
  AckRefugeResistantToB[0] := 0;
  AckRefugeResistantToAB[0] := 0;
  AckRefugeCells[0] := 0;

  {Set X scale legend}

  if (RunningTimeHour*60) + RunningTimeMinute < 120 then
    NewForm.Chart1.AxisList.BottomAxis.Title.Caption := 'Time (minutes)'
  else NewForm.Chart1.AxisList.BottomAxis.Title.Caption := 'Time (hours)';

  {Set Y scale}

  if Form1.LogScale.Checked = True then
  begin
    DoLog := true;
    NewForm.Chart1.AxisList[0].Marks.Format := '%2f';
    NewForm.Chart1.AxisList[0].Title.Caption := 'log10 CFU, PFU /ml';
  end
  else
  begin
    DoLog := false;
    NewForm.Chart1.AxisList[0].Marks.Format := '%:8.3e';
    NewForm.Chart1.AxisList[0].Title.Caption := 'CFU, PFU /ml';
    NewForm.LeftAxisAutoScaleTransform.Enabled := True;
  end;

  {Check output selected}

  if (SensitiveOut.Checked = False) and (ResistantAOut.Checked = False) and (ResistantBOut.Checked = False) and (ResistantABOut.Checked = False)
    and (ResourceCOut.Checked = False) and (PhageAOut.Checked = False) and (PhageBOut.Checked = False) and (InfectedAOut.Checked = False) and
    (InfectedBOut.Checked = False) and (InfectedABOut.Checked = False) and (AResInfbyBOut.Checked = False) and (BResInfbyAOut.Checked = False) and
    (RefugeOut.Checked = False)
    then NewForm.StatusBar1.SimpleText := 'No output parameter selected!' else Form1.StatusBar1.SimpleText := 'Running..';

    OutputParam := ('Output parameters: ');
    if SensitiveOut.Checked = true then OutputParam := OutputParam + (' 1 '); if ResistantAOut.Checked = true then OutputParam := OutputParam + (' 2 ');
    if ResistantBOut.Checked = true then OutputParam := OutputParam + (' 3 '); if ResistantABOut.Checked = true then OutputParam := OutputParam + (' 4 ');
    if InfectedAOut.Checked = true then OutputParam := OutputParam + (' 5 '); if InfectedBOut.Checked = true then OutputParam := OutputParam + (' 6 ');
    if InfectedABOut.Checked = true then OutputParam := OutputParam + (' 7 '); if AResInfbyBOut.Checked = true then OutputParam := OutputParam + (' 8 ');
    if BResInfbyAOut.Checked = true then OutputParam := OutputParam + (' 9 '); if RefugeOut.Checked = true then OutputParam := OutputParam + (' 10 ');
    if PhageAOut.Checked = true then OutputParam := OutputParam + (' 11 '); if PhageBOut.Checked = true then OutputParam := OutputParam + (' 12 ');
    if ResourceCOut.Checked = true then OutputParam := OutputParam + (' 13 '); if LogScale.Checked = true then OutputParam := OutputParam + (' 14 ');
    if RoundOff.Checked = true then OutputParam := OutputParam + (' 15 ');

  {Convert clock time to time in step size = RunningTime}

  RunningTime := (RunningTimeHour*HourDivisor) + (RunningTimeMinute*MinuteDivisor);

  for TimeStep := 0 to RunningTime do                         {TimeStep = Main loop}
    begin
      if (RunningTimeHour*60) + RunningTimeMinute < 120 then TimeOut := TimeStep/MinuteDivisor
      else TimeOut := TimeStep/HourDivisor;

      if (TimeStep Mod 10000) = 1 then Form1.StatusBar1.SimpleText := Form1.StatusBar1.SimpleText + '.';
      if (Length(Form1.StatusBar1.SimpleText) > 12) then Form1.StatusBar1.SimpleText := 'Running.';
      Form1.StatusBar1.Update;

    {Adding phages at different times}

    if TimeStep = AddTimeOneA then FreePhagesA := PhageTitreA;
    if TimeStep = AddTimeOneB then FreePhagesB := PhageTitreB;
    if (AddTimeTwoA > AddTimeOneA) and (TimeStep = AddTimeTwoA) then FreePhagesA := FreePhagesA + PhageTitreA;
    if (AddTimeTwoB > AddTimeOneB) and (TimeStep = AddTimeTwoB) then FreePhagesB := FreePhagesB + PhageTitreB;
    if (AddTimeThreeA > AddTimeOneA) and (AddTimeThreeA > AddTimeTwoA) and (TimeStep = AddTimeThreeA) then FreePhagesA := FreePhagesA + PhageTitreA;
    if (AddTimeThreeB > AddTimeOneB) and (AddTimeThreeB > AddTimeTwoB) and (TimeStep = AddTimeThreeB) then FreePhagesB := FreePhagesB + PhageTitreB;

    {Write lines, start with time 0}

    if SensitiveOut.Checked = True then NewForm.Chart1SensitiveOutLineSeries1.AddXY(TimeOut,GetLineNumbers(Sensitive));
    if ResistantAOut.Checked = True then NewForm.Chart1ResistantAOutLineSeries2.AddXY(TimeOut,GetLineNumbers(ResistantToA));
    if ResistantBOut.Checked = True then NewForm.Chart1ResistantBOutLineSeries3.AddXY(TimeOut,GetLineNumbers(ResistantToB));
    if ResistantABOut.Checked = True then NewForm.Chart1ResistantABOutLineSeries4.AddXY(TimeOut,GetLineNumbers(ResistantToAB));
    if ResourceCOut.Checked = True then
      begin
        NewForm.Chart1.AxisList[2].Visible := True;
        NewForm.Chart1ResourceCOutLineSeries5.AddXY(TimeOut,ResourceC);
      end
      else NewForm.Chart1.AxisList[2].Visible := False;
    if PhageAOut.Checked = True then NewForm.Chart1PhageAOutLineSeries6.AddXY(TimeOut,GetLineNumbers(FreePhagesA));
    if PhageBOut.Checked = True then NewForm.Chart1PhageBOutLineSeries7.AddXY(TimeOut,GetLineNumbers(FreePhagesB));
    if InfectedAOut.Checked = True then NewForm.Chart1InfectedAOutLineSeries8.AddXY(TimeOut,GetLineNumbers(InfectedA));
    if InfectedBOut.Checked = True then NewForm.Chart1InfectedBOutLineSeries9.AddXY(TimeOut,GetLineNumbers(InfectedB));
    if InfectedABOut.Checked = True then NewForm.Chart1InfectedABOutLineSeries10.AddXY(TimeOut,GetLineNumbers(InfectedAB));
    if AResInfbyBOut.Checked = True then NewForm.Chart1AResInfBOutLineSeries11.AddXY(TimeOut,GetLineNumbers(AResInfectedB));
    if BResInfbyAOut.Checked = True then NewForm.Chart1BResInfAOutLineSeries12.AddXY(TimeOut,GetLineNumbers(BResInfectedA));
    if RefugeOut.Checked = True then NewForm.Chart1RefugeOutLineSeries13.AddXY(TimeOut,GetLineNumbers(TotalRefugeCells));
      {Calculate bacterial growth, decay and outflow}

    ResourceC := ResourceC + ((ResevoirC-ResourceC) * FlowRate);

    NewSensitive := Sensitive * MaxGrowthSens * MonodFunc;
    Sensitive := Sensitive + NewSensitive - (DecayRateBact * Sensitive) - (FlowRate * Sensitive);
    NewResistantToA := ResistantToA * MaxGrowthResA * MonodFunc;
    ResistantToA := ResistantToA + NewResistantToA - (DecayRateBact * ResistantToA) - (FlowRate * ResistantToA);
    NewResistantToB := ResistantToB * MaxGrowthResB * MonodFunc;
    ResistantToB := ResistantToB + NewResistantToB - (DecayRateBact * ResistantToB) - (FlowRate * ResistantToB);
    NewResistantToAB := ResistantToAB * MaxGrowthResAB * MonodFunc;
    ResistantToAB := ResistantToAB + NewResistantToAB - (DecayRateBact * ResistantToAB) - (FlowRate * ResistantToAB);

    ResourceC := ResourceC - (Sensitive * ConversEff * MaxGrowthSens * MonodFunc);
    ResourceC := ResourceC - (ResistantToA * ConversEff * MaxGrowthResA * MonodFunc);
    ResourceC := ResourceC - (ResistantToB * ConversEff * MaxGrowthResB * MonodFunc);
    ResourceC := ResourceC - (ResistantToAB * ConversEff * MaxGrowthResAB * MonodFunc);

    if ResourceC < 0 then ResourceC := 0; {Probably not needed}

    MonodFunc := ResourceC/(ResourceC+MonodK);

    InfectedA := InfectedA - (FlowRate*InfectedA) - (DecayRateBact*InfectedA);
    InfectedB := InfectedB - (FlowRate*InfectedB) - (DecayRateBact*InfectedB);
    InfectedAB := InfectedAB - (FlowRate*InfectedAB) - (DecayRateBact*InfectedAB);
    AResInfectedB := AResInfectedB - (FlowRate*AResInfectedB) - (DecayRateBact*AResInfectedB);
    BResInfectedA := BResInfectedA - (FlowRate*BResInfectedA) - (DecayRateBact*BResInfectedA);
   {End bacterial growth / outflow}

     {Mutation of bacteria, two models: Deterministic and stochastic}
     case rgMutationMod.ItemIndex of
       0 : begin {Flat deterministic mutations}
         SumOfNewSensitiveMutations := 0;
         NewMutations := 0;
         if NewSensitive > 1 then

          begin
            NewMutations := NewSensitive * ResistRateA;
            ResistantToA := ResistantToA + NewMutations;
            SumOfNewSensitiveMutations := SumOfNewSensitiveMutations + NewMutations;

            NewMutations := NewSensitive * ResistRateB;
            ResistantToB := ResistantToB + NewMutations;
            SumOfNewSensitiveMutations := SumOfNewSensitiveMutations + NewMutations;

            NewMutations := NewSensitive * ResistRateAB;
            ResistantToAB := ResistantToAB + NewMutations;
            SumOfNewSensitiveMutations := SumOfNewSensitiveMutations + NewMutations;
            Sensitive := Sensitive - SumOfNewSensitiveMutations;
           end;
         if NewResistantToA > 1 then
          begin
            NewMutations := NewResistantToA * ResistRateB;
            ResistantToAB := ResistantToAB + NewMutations;
            ResistantToA := ResistantToA - NewMutations;
           end;
         if NewResistantToB > 1 then
          begin
            NewMutations := NewResistantToB * ResistRateA;
            ResistantToAB := ResistantToAB + NewMutations;
            ResistantToB := ResistantToB - NewMutations;
          end;
       end;

     1 : begin {Random mutations}
         SumOfNewSensitiveMutations := 0;
         NewMutations := 0;
         if NewSensitive > 1 then
          begin
           NewMutations := MutStochastic(NewSensitive, ResistRateA);
           ResistantToA := ResistantToA + NewMutations;
           SumOfNewSensitiveMutations := SumOfNewSensitiveMutations + NewMutations;

           NewMutations := MutStochastic(NewSensitive, ResistRateB);
           ResistantToB := ResistantToB + NewMutations;
           SumOfNewSensitiveMutations := SumOfNewSensitiveMutations + NewMutations;

           NewMutations := MutStochastic(NewSensitive, ResistRateAB);
           ResistantToAB := ResistantToAB + NewMutations;
           SumOfNewSensitiveMutations := SumOfNewSensitiveMutations + NewMutations;
           Sensitive := Sensitive - SumOfNewSensitiveMutations;
          end;
         if NewResistantToA > 1 then
          begin
           NewMutations := MutStochastic(NewResistantToA, ResistRateB);
           ResistantToAB := ResistantToAB + NewMutations;
           ResistantToA := ResistantToA - NewMutations;
          end;
         if NewResistantToB > 1 then
          begin
           NewMutations := MutStochastic(NewResistantToB, ResistRateA);
           ResistantToAB := ResistantToAB + NewMutations;
           ResistantToB := ResistantToB - NewMutations;
          end;
       end;
     end; {End Case mutation modes}

       if ResistantToA < 0 then  ResistantToA := 0;
       if ResistantToB < 0 then  ResistantToB := 0;
       if ResistantToAB < 0 then  ResistantToAB := 0;

 {Calculate refuge cells}

  if (RefugeRateIn > 0) and (RefugeRateOut > 0) then
  begin
    AllSusceptible := Sensitive + ResistantToA + ResistantToB + ResistantToAB;

    case rgRefugeCells.ItemIndex of
     0: begin
          IRefugePopNr := 0;
          TotalRefugeRateOut := RefugeRateOut;
          CellsToRefuge;
          CellsFromRefuge;
          if AckRefugeCells[IRefugePopNr] > 0 then
            begin
              AckRefugeSensitive[IRefugePopNr] := AckRefugeSensitive[IRefugePopNr] - (DecayRateBact * AckRefugeSensitive[IRefugePopNr]) - (FlowRate * AckRefugeSensitive[IRefugePopNr]);
              AckRefugeResistantToA[IRefugePopNr] := AckRefugeResistantToA[IRefugePopNr] - (DecayRateBact * AckRefugeResistantToA[IRefugePopNr]) - (FlowRate * AckRefugeResistantToA[IRefugePopNr]);
              AckRefugeResistantToB[IRefugePopNr] := AckRefugeResistantToB[IRefugePopNr] - (DecayRateBact * AckRefugeResistantToB[IRefugePopNr]) - (FlowRate * AckRefugeResistantToB[IRefugePopNr]);
              AckRefugeResistantToAB[IRefugePopNr] := AckRefugeResistantToAB[IRefugePopNr] - (DecayRateBact * AckRefugeResistantToAB[IRefugePopNr]) - (FlowRate * AckRefugeResistantToAB[IRefugePopNr]);
              AckRefugeCells[IRefugePopNr] := AckRefugeSensitive[IRefugePopNr] +  AckRefugeResistantToA[IRefugePopNr] + AckRefugeResistantToB[IRefugePopNr] + AckRefugeResistantToAB[IRefugePopNr];
            end;
          TotalRefugeCells := AckRefugeCells[IRefugePopNr];
        end;

     1: begin
          if AllSusceptible * RefugeRateIn > 10 then             {Max RefugeRateIn = 0.01}
            begin
              Inc(IRefugePopNr);
              SetLength(AckRefugeSensitive, 1 + IRefugePopNr);
              SetLength(AckRefugeResistantToA, 1 + IRefugePopNr);
              SetLength(AckRefugeResistantToB, 1 + IRefugePopNr);
              SetLength(AckRefugeResistantToAB, 1 + IRefugePopNr);
              SetLength(AckRefugeCells, 1 + IRefugePopNr);
            end;
        CellsToRefuge;

        TotalRefugeCells := 0;
        for I := 0 to IRefugePopNr do TotalRefugeCells := TotalRefugeCells + AckRefugeCells[I];
        TotalRefugeRateOut := TotalRefugeCells * RefugeRateOut / AckRefugeCells[IRefugePopNr];
        if IRefugePopNr > 0 then
          begin
            while TotalRefugeRateOut > 1 do
              begin
                AckRefugeCells[IRefugePopNr-1] := AckRefugeCells[IRefugePopNr-1] + AckRefugeCells[IRefugePopNr];
                Dec(IRefugePopNr);
                SetLength(AckRefugeSensitive, 1 + IRefugePopNr);
                SetLength(AckRefugeResistantToA, 1 + IRefugePopNr);
                SetLength(AckRefugeResistantToB, 1 + IRefugePopNr);
                SetLength(AckRefugeResistantToAB, 1 + IRefugePopNr);
                SetLength(AckRefugeCells, 1 + IRefugePopNr);
                TotalRefugeRateOut := TotalRefugeCells * RefugeRateOut / AckRefugeCells[IRefugePopNr];
              end;
            end;
        CellsFromRefuge;
        end;
      end; {case rgRefugeCells.ItemIndex}
  end; {End of calculate refuge cells}

     {Start Infection by Phages}

     if PhageInfection then
     begin

       TotalBacteria := Sensitive + ResistantToA + ResistantToB + ResistantToAB +       {Set population sizes for infection}
         InfectedA + InfectedB + InfectedAB + AResInfectedB + BResInfectedA;
       UnInfectedA := Sensitive+InfectedB+ResistantToB;
       UnInfectedB := Sensitive+InfectedA+ResistantToA;
       AllInfectedA := InfectedA + InfectedAB + BResInfectedA;
       AllInfectedB := InfectedB + InfectedAB + AResInfectedB;
       FreePhagesA := FreePhagesA-(FreePhagesA*FlowRate)-(FreePhagesA*DecayRatePhageA);
       FreePhagesB := FreePhagesB-(FreePhagesB*FlowRate)-(FreePhagesB*DecayRatePhageB);
       if (rgRefugeCells.ItemIndex = 0) and (RefugeRateIn > 0) and (RefugeRateOut > 0) then
         begin
           AllInfectedA := AllInfectedA + AckRefugeSensitive[0] + AckRefugeResistantToB[0];
           AllInfectedB := AllInfectedB + AckRefugeSensitive[0] + AckRefugeResistantToA[0];
         end;
      case rgInfectionMod.ItemIndex of
      0 : begin                                                                 {Standard mass action mode of infection}
        if TotalBacteria > 0 then
         begin
           if rgAdsorptionMod.ItemIndex = 1 then BoundPhagesA := (AdsorbRateA*(UnInfectedA+AllInfectedA)*FreePhagesA) else BoundPhagesA := (AdsorbRateA*UnInfectedA*FreePhagesA);
           if BoundPhagesA > FreePhagesA then BoundPhagesA := FreePhagesA;
           if BoundPhagesA > UnInfectedA then BoundPhagesA := UnInfectedA;
           if (AdsorbRateA*UnInfectedA*FreePhagesA) > 0 then FreqNewInfectedA := (AdsorbRateA*UnInfectedA*FreePhagesA) / TotalBacteria else FreqNewInfectedA := 0;
           FreePhagesA := FreePhagesA - BoundPhagesA;
           if rgAdsorptionMod.ItemIndex = 1 then BoundPhagesB := (AdsorbRateB*(UnInfectedB+AllInfectedB)*FreePhagesB) else BoundPhagesB := (AdsorbRateB*UnInfectedB*FreePhagesB);
           if BoundPhagesB > FreePhagesB then BoundPhagesB := FreePhagesB;
           if BoundPhagesB > UnInfectedB then BoundPhagesB := UnInfectedB;
           if (AdsorbRateB*UnInfectedB*FreePhagesB) > 0 then FreqNewInfectedB := (AdsorbRateB*UnInfectedB*FreePhagesB) / TotalBacteria else FreqNewInfectedB := 0;
           FreePhagesB := FreePhagesB - BoundPhagesB;
        end;
      end;                                                                      {End standard mass action mode of infection}

      1 : begin
       if TotalBacteria > 0 then                                                {Poisson mode of infection}
        begin
          BoundPhagesA :=  (1-(Exp(-AdsorbRateA*UnInfectedA)))*FreePhagesA;
          if UnInfectedA > 0 then FreqNewInfectedA :=  (1-(Exp(-BoundPhagesA/UnInfectedA)))*UnInfectedA/TotalBacteria else FreqNewInfectedA := 0;
          if rgAdsorptionMod.ItemIndex = 1 then BoundPhagesA :=  (1-(Exp(-AdsorbRateA*(UnInfectedA+AllInfectedA))))*FreePhagesA;
          FreePhagesA := FreePhagesA - BoundPhagesA;
          BoundPhagesB :=  (1-(Exp(-AdsorbRateB*UnInfectedB)))*FreePhagesB;
          if UnInfectedB > 0 then FreqNewInfectedB :=  (1-(Exp(-BoundPhagesB/UnInfectedB)))*UnInfectedB/TotalBacteria else FreqNewInfectedB := 0;
          if rgAdsorptionMod.ItemIndex = 1 then BoundPhagesB :=  (1-(Exp(-AdsorbRateB*(UnInfectedB+AllInfectedB))))*FreePhagesB;
          FreePhagesB := FreePhagesB - BoundPhagesB;
        end;
      end;                                                                      {End Poisson mode of infection}
     end; {End Case: Infection modes}

     if FreePhagesA < 0 then FreePhagesA := 0;
     if FreePhagesB < 0 then FreePhagesB := 0;
     if FreqNewInfectedA < 0 then FreqNewInfectedA := 0;
     if FreqNewInfectedB < 0 then FreqNewInfectedB := 0;

     {Frequencies of classes of bacteria to be infected}
     if (Sensitive + ResistantToB + InfectedB) > 1 then
      begin
       FreqNewSensInfectedA := Sensitive / (Sensitive+InfectedB+ResistantToB) * FreqNewInfectedA;
       FreqNewBResInfectedA := ResistantToB /(Sensitive+InfectedB+ResistantToB) * FreqNewInfectedA;
       FreqNewInfectedAB1 := InfectedB / (Sensitive+InfectedB+ResistantToB) * FreqNewInfectedA;
      end;

      if (Sensitive + ResistantToA + InfectedA) > 1 then
      begin
       FreqNewSensInfectedB := Sensitive / (Sensitive+InfectedA+ResistantToA) * FreqNewInfectedB;
       FreqNewAResInfectedB := ResistantToA / (Sensitive+InfectedA+ResistantToA) * FreqNewInfectedB;
       FreqNewInfectedAB2 := InfectedA / (Sensitive+InfectedA+ResistantToA) * FreqNewInfectedB;
      end; {End frequencies of classes of bacteria to be infected}

      {Calculate newly and accumulated infected bacteria and start conveyors}

      {Calculate Susceptible bacteria Infected by A, Infected by B and Infected by A and B}

      if (FreqNewSensInfectedA > 0) and (FreqNewSensInfectedB > 0) then
      begin
       FreqNewSensInfectedAB := FreqNewSensInfectedA * FreqNewSensInfectedB;
       FreqNewSensInfectedA := FreqNewSensInfectedA - FreqNewSensInfectedAB;
       FreqNewSensInfectedB := FreqNewSensInfectedB - FreqNewSensInfectedAB;
       NewInfectedAB3 := FreqNewSensInfectedAB * TotalBacteria;
       InfectedAB3 := InfectedAB3 + NewInfectedAB3;
       Sensitive := Sensitive - NewInfectedAB3;
      end else NewInfectedAB3 := 0;

      if FreqNewSensInfectedA > 0 then
       begin
         NewInfectedA := FreqNewSensInfectedA * TotalBacteria;
         InfectedA := InfectedA + NewInfectedA;
         Sensitive := Sensitive - NewInfectedA;
       end else NewInfectedA := 0;

      if FreqNewSensInfectedB > 0 then
       begin
         NewInfectedB := FreqNewSensInfectedB * TotalBacteria;
         InfectedB := InfectedB + NewInfectedB;
         Sensitive := Sensitive - NewInfectedB;
       end else NewInfectedB := 0;

       if Sensitive < 0 then Sensitive := 0;

       if InfectedA > 0 then
       begin
         ConveyorA[ITimeA] := NewInfectedA;
         ITimeToLysisA[ITimeA] := LatencyA - ITimeA;
         if ITimeA < (LatencyA-1) then Inc(ITimeA) else
         begin
           ITimeA := 0;
           ReleaseA := true;
         end;
         if ReleaseA then
         begin
           InfectedA := InfectedA - ConveyorA[ITimeA];
           if InfectedA < 0 then InfectedA := 0;
           FreePhagesA := FreePhagesA + (ConveyorA[ITimeA] * BurstA);
         end;
      end
       else
       begin
         ITimeA := 0;
         ReleaseA := false;
       end;

       if InfectedB > 0 then
       begin
         ConveyorB[ITimeB] := NewInfectedB;
         ITimeToLysisB[ITimeB] := LatencyB - ITimeB;
         if ITimeB < (LatencyB-1) then Inc(ITimeB) else
         begin
           ITimeB := 0;
           ReleaseB := true;
         end;
         if ReleaseB then
         begin
           InfectedB := InfectedB - ConveyorB[ITimeB];
           if InfectedB < 0 then InfectedB := 0;
           FreePhagesB := FreePhagesB + (ConveyorB[ITimeB] * BurstB);
         end;
       end
       else
       begin
         ITimeB := 0;
         ReleaseB := false;
       end;

     {End of calculating Susceptible bacteria Infected by A, Infected by B and Infected by A and B}

     {Calculate Infected bacteria infected by the other: InfectedA by B, InfectedB by A}

       if (FreqNewInfectedAB1 > 0) and (InfectedB > 0) then
       begin
        NewInfectedAB1 := FreqNewInfectedAB1 * TotalBacteria;
        InfectedAB1 := InfectedAB1  + NewInfectedAB1;
        for ICount := 0 to LatencyB-1 do
        begin
          if ITimeToLysisB[ICount] > LatencyA then
           NewInfectedAonBA := NewInfectedAonBA  + (NewInfectedAB1 * ConveyorB[ICount] / InfectedB)                   {For killing of AB and release of phage A}
            else ConveyorAonBB[ICount] := ConveyorAonBB[ICount] + (NewInfectedAB1 * ConveyorB[ICount] / InfectedB);   {For killing of AB and release of phage B}
          ConveyorB[ICount] := ConveyorB[ICount] - (NewInfectedAB1 * ConveyorB[ICount] / InfectedB);                  {Remaining uninfected by A (release of phage B)}
        end;
        InfectedB := InfectedB - NewInfectedAB1;
        end
	else  NewInfectedAB1 := 0;

       if (FreqNewInfectedAB2 > 0) and (InfectedA > 0) then
       begin
        NewInfectedAB2 := FreqNewInfectedAB2 * TotalBacteria;
        InfectedAB2 := InfectedAB2  + NewInfectedAB2;
        for ICount := 0 to LatencyA-1 do
        begin
          if ITimeToLysisA[ICount] > LatencyB then
             NewInfectedBonAB := NewInfectedBonAB  + (NewInfectedAB2 * ConveyorA[ICount] / InfectedA)                 {For killing of AB and release of phage B}
            else ConveyorBonAA[ICount] := ConveyorBonAA[ICount] + (NewInfectedAB2 * ConveyorA[ICount] / InfectedA);   {For killing of AB and release of phage A}
          ConveyorA[ICount] := ConveyorA[ICount] - (NewInfectedAB2 * ConveyorA[ICount] / InfectedA);                  {Remaining uninfected by B (release of phage A)}
        end;
        InfectedA := InfectedA - NewInfectedAB2;
        end
	else  NewInfectedAB2 := 0;

     if InfectedAB1 > 0 then
     begin
       ConveyorAonBA[ITimeABA] := NewInfectedAonBA;
       if ITimeABA < (LatencyA-1) then Inc(ITimeABA) else
         begin
          ITimeABA := 0;
          ReleaseABA := true;
         end;
         if ReleaseABA and (ConveyorAonBA[ITimeABA] > 0)then
         begin
           InfectedAB1 := InfectedAB1 - ConveyorAonBA[ITimeABA];
           if InfectedAB1 < 1 then InfectedAB1 := 0;
           FreePhagesA := FreePhagesA + (ConveyorAonBA[ITimeABA] * BurstA);
         end;
     end
     else
       begin
         ITimeABA := 0;
         ReleaseABA := false;
       end;

     if ReleaseB then ReleaseABB := ReleaseB;

     if InfectedAB1 > 0 then
       begin
         if ITimeABB < (LatencyB-1) then Inc(ITimeABB) else
           begin
             ITimeABB := 0;
             ReleaseABB := true;
           end;
         if ReleaseABB and (ConveyorAonBB[ITimeABB] > 0) then
          begin
            InfectedAB1 := InfectedAB1 - ConveyorAonBB[ITimeABB];
            if InfectedAB1 < 1 then InfectedAB1 := 0;
            FreePhagesB := FreePhagesB + (ConveyorAonBB[ITimeABB] * BurstB);
          end;
       end
        else
       begin
         ITimeABB := 0;
         ReleaseABB := false;
       end;

     if InfectedAB2 > 0 then
     begin
       ConveyorBonAB[ITimeBAB] := NewInfectedBonAB;
       if ITimeBAB < (LatencyB-1) then Inc(ITimeBAB) else
         begin
          ITimeBAB := 0;
          ReleaseBAB := true;
         end;
         if ReleaseBAB and (ConveyorBonAB[ITimeBAB] > 0) then
         begin
           InfectedAB2 := InfectedAB2 - ConveyorBonAB[ITimeBAB];
           if InfectedAB2 < 1 then InfectedAB2 := 0;
           FreePhagesB := FreePhagesB + (ConveyorBonAB[ITimeBAB] * BurstB);
         end;
     end
      else
       begin
         ITimeBAB := 0;
         ReleaseBAB := false;
       end;

     if ReleaseA then ReleaseBAA := ReleaseA;

      if InfectedAB2 > 0 then
        begin
        if ITimeBAA < (LatencyA-1) then Inc(ITimeBAA) else
         begin
          ITimeBAA := 0;
          ReleaseBAA := true;
         end;
         if ReleaseBAA and (ConveyorBonAA[ITimeBAA] > 0) then
           begin
            InfectedAB2 := InfectedAB2 - ConveyorBonAA[ITimeBAA];
            if InfectedAB2 < 1 then InfectedAB2 := 0;
            FreePhagesA := FreePhagesA + (ConveyorBonAA[ITimeBAA] * BurstA);
           end;
        end
      else
       begin
         ITimeBAA := 0;
         ReleaseBAA := false;
       end;

      {End of calculating Infected bacteria infected by the other: InfectedA by B, InfectedB by A}

      {Calculate Resistant bacteria infected by the other A Resistant infected by B, B resistant infected by A}

       if FreqNewBResInfectedA > 0 then
        begin
         NewBResInfectedA := FreqNewBResInfectedA * TotalBacteria;
         BResInfectedA := BResInfectedA + NewBResInfectedA;
         ResistantToB := ResistantToB - NewBResInfectedA;
        end else NewBResInfectedA := 0;

       if FreqNewAResInfectedB > 0 then
        begin
         NewAResInfectedB := FreqNewAResInfectedB * TotalBacteria;
         AResInfectedB := AResInfectedB + NewAResInfectedB;
         ResistantToA := ResistantToA - NewAResInfectedB;
        end else NewAResInfectedB := 0;

     if BResInfectedA > 0 then
     begin
       ConveyorBResA[ITimeBResA] := NewBResInfectedA;
       if ITimeBResA < (LatencyA-1) then Inc(ITimeBResA) else
       begin
        ITimeBResA := 0;
        ReleaseBResA := true;
       end;
       if ReleaseBResA then
       begin
         ResistantToB := ResistantToB - ConveyorBResA[ITimeBResA];
         if ResistantToB < 0 then ResistantToB := 0;
         FreePhagesA := FreePhagesA + (ConveyorBResA[ITimeBResA] * BurstA);
       end;
     end
     else
       begin
         ITimeBResA := 0;
         ReleaseBResA := false;
       end;

      if AResInfectedB > 0 then
     begin
       ConveyorAResB[ITimeAResB] := NewAResInfectedB;
       if ITimeAResB < (LatencyB-1) then Inc(ITimeAResB) else
       begin
        ITimeAResB := 0;
        ReleaseAResB := true;
       end;
       if ReleaseAResB then
       begin
         ResistantToA := ResistantToA - ConveyorAResB[ITimeAResB];
         if ResistantToA < 0 then ResistantToA := 0;
         FreePhagesB := FreePhagesB + (ConveyorAResB[ITimeAResB] * BurstB);
       end;
     end
      else
       begin
         ITimeAResB := 0;
         ReleaseAResB := false;
       end;

     {End of calculating Resistant bacteria infected by the other A Resistant infected by B, B resistant infected by A}

     {Calculate and sum up all bacteria infected by both phages}

     if InfectedAB3 > 0 then
     begin
       ConveyorAB[ITimeAB] := NewInfectedAB3;
       if ITimeAB < (LatencyAB-1) then Inc(ITimeAB) else
       begin
        ITimeAB := 0;
        ReleaseAB := true;
       end;
       if ReleaseAB then
       begin
         InfectedAB3 := InfectedAB3 - ConveyorAB[ITimeAB];
         if InfectedAB3 < 1 then InfectedAB3 := 0;
         if LatencyA < LatencyB then FreePhagesA := FreePhagesA + (ConveyorAB[ITimeAB] * BurstA)
         else if LatencyB < LatencyA then FreePhagesB := FreePhagesB + (ConveyorAB[ITimeAB] * BurstB)
         else
           begin
            FreePhagesA := FreePhagesA + (ConveyorAB[ITimeAB] * BurstA/2);
            FreePhagesB := FreePhagesB + (ConveyorAB[ITimeAB] * BurstB/2);
           end;
       end;
     end
     else
       begin
         ITimeAB := 0;
         ReleaseAB := false;
       end;

     InfectedAB := InfectedAB1 + InfectedAB2 + InfectedAB3;
     UnInfectedA := Sensitive+InfectedB+ResistantToB;
     UnInfectedB := Sensitive+InfectedA+ResistantToA;
     if UnInfectedA < 0 then UnInfectedA := 0;
     if UnInfectedB < 0 then UnInfectedB := 0;

     {End of calculating and sum up all bacteria infected by both phages}

       FreqNewSensInfectedA := 0;
       FreqNewInfectedAB1 := 0;
       FreqNewBResInfectedA := 0;
       FreqNewSensInfectedB := 0;
       FreqNewInfectedAB2 := 0;
       FreqNewAResInfectedB := 0;
       FreqNewSensInfectedAB := 0;
       FreqNewInfectedAB := 0;
       NewInfectedAonBA := 0;
       NewInfectedBonAB := 0;

       {End of calculating newly and accumulated infected bacteria and start conveyors}
     if InfectedA < 0 then InfectedA := 0;
     if InfectedB < 0 then InfectedB := 0;
     if InfectedAB < 0 then InfectedAB := 0;
     if AResInfectedB < 0 then AResInfectedB := 0;
     if BResInfectedA < 0 then BResInfectedA := 0;
     if TotalRefugeCells < 0 then TotalRefugeCells := 0;

     if RoundOff.Checked = True then
     begin
       if Sensitive < 1 then Sensitive := 0;
       if ResistantToA < 1 then ResistantToA := 0;
       if ResistantToB < 1 then ResistantToB := 0;
       if ResistantToAB < 1 then ResistantToAB := 0;
       if InfectedA < 1 then InfectedA := 0;
       if InfectedB < 1 then InfectedB := 0;
       if InfectedAB < 1 then InfectedAB := 0;
       if AResInfectedB < 1 then AResInfectedB := 0;
       if BResInfectedA < 1 then BResInfectedA := 0;
       if TotalRefugeCells < 1 then TotalRefugeCells := 0;
       if FreePhagesA < 1 then FreePhagesA := 0;
       if FreePhagesB < 1 then FreePhagesB := 0;
     end;

     end; {End infection by phages}

  end; {TimeStep = main loop}

  {Set graph extent and allow mouse operations}
  if RunningTime < 90 then NewForm.Chart1.ExtentSizeLimit.XMax := RunningTime
  else NewForm.Chart1.ExtentSizeLimit.XMax := RunningTime /60;
  NewForm.Chart1.ExtentSizeLimit.XMin := 0;
  NewForm.Chart1.ExtentSizeLimit.YMin := 0;

   {Show the form and graph}
   NewForm.Show;
   Form1.StatusBar1.SimpleText := ' ';
end; {TForm1.Button1Click}


procedure TForm1.Button2Click(Sender: TObject);
begin
  Close;
end;

procedure TForm1.Button3Click(Sender: TObject);
begin
  if OpenDialog1.Execute then
    begin
      filename := OpenDialog1.FileName;
      ReadCTLFile;
    end;
    Form1.Button1.SetFocus;
end;

procedure TForm1.Button4Click(Sender: TObject);
begin
  if FileExists ('Cocktail.pdf') then OpenDocument('Cocktail.pdf') else
    ShowMessage('Cannot find the Cocktail info document. It must'
    + sLineBreak + 'be placed in the same folder as the program');
end;

procedure ReadCTLFile;
var
  TempString: string;
  DataStrings: TStringList;
begin
  ClearOldSet;
  DataStrings := TStringList.Create;
  try
    DataStrings.LoadFromFile(filename);
        if Pos('= Cocktail input file',DataStrings.Strings[0]) <> 0 then
        begin
          with Form1 do
          begin
          DefaultFormatSettings.DecimalSeparator:= '.';
          FloatSpinEdit1.Value := StrToFloat(Copy(DataStrings.Strings[3],30));    {Max growth rate}
          FloatSpinEdit2.Value := StrToFloat(Copy(DataStrings.Strings[4],18));    {Monod const}
          Edit1.Text := Copy(DataStrings.Strings[5],25);                          {Conversion eff}
          FloatSpinEdit3.Value := StrToFloat(Copy(DataStrings.Strings[6],24));    {Bact decay rate}
          Edit2.Text := Copy(DataStrings.Strings[7],30);                          {Resistant to A rate}
          Edit3.Text := Copy(DataStrings.Strings[8],30);                          {Resistant to B rate}
          Edit4.Text := Copy(DataStrings.Strings[9],25);                          {Resistant freq to A}
          Edit5.Text := Copy(DataStrings.Strings[10],25);                         {Resistant freq to B}
          Edit11.Text := Copy(DataStrings.Strings[11],26);                        {Resistant freq to AB}
          FloatSpinEdit7.Value := StrToFloat(Copy(DataStrings.Strings[12],31));   {Max growth rate A resistant}
          FloatSpinEdit8.Value := StrToFloat(Copy(DataStrings.Strings[13],31));   {Max growth rate B resistant}
          FloatSpinEdit9.Value := StrToFloat(Copy(DataStrings.Strings[14],32));   {Max growth rate AB resistant}
          Edit8.Text := Copy(DataStrings.Strings[15],15);                         {Bacterial start titre}
          SpinEdit1.Value := StrToInt(Copy(DataStrings.Strings[18],23));        {Resource start C}
          SpinEdit2.Value := StrToInt(Copy(DataStrings.Strings[19],26));        {Resource resevoir C}
          FloatSpinEdit4.Value := StrToFloat(Copy(DataStrings.Strings[20],13));   {Max growth rate AB resistant}
          Edit9.Text := Copy(DataStrings.Strings[23],22,Pos(',',DataStrings.Strings[23])-22);            {Adsorption rate A}
          Edit10.Text := Copy(DataStrings.Strings[23],Pos(',',DataStrings.Strings[23])+6);               {Adsorption rate B}
          SpinEdit4.Value := StrToInt(Copy(DataStrings.Strings[24],19,Pos(',',DataStrings.Strings[24])-19));       {Latency time A}
          SpinEdit10.Value := StrToInt(Copy(DataStrings.Strings[24],Pos(',',DataStrings.Strings[24])+6));          {Latency time B}
          SpinEdit5.Value := StrToInt(Copy(DataStrings.Strings[25],17,Pos(',',DataStrings.Strings[25])-17));       {Burst A}
          SpinEdit11.Value := StrToInt(Copy(DataStrings.Strings[25],Pos(',',DataStrings.Strings[25])+6));          {Burst B}
          FloatSpinEdit5.Value := StrToFloat(Copy(DataStrings.Strings[26],17,Pos(',',DataStrings.Strings[26])-17));  {Decay A}
          FloatSpinEdit6.Value := StrToFloat(Copy(DataStrings.Strings[26],Pos(',',DataStrings.Strings[26])+6));      {Decay B}
          Edit6.Text := Copy(DataStrings.Strings[27],18,Pos(',',DataStrings.Strings[27])-18);  {Added titre A}
          Edit7.Text := Copy(DataStrings.Strings[27],Pos(',',DataStrings.Strings[27])+6);      {Added titre B}
          TempString := DataStrings.Strings[28];
          SpinEdit7.Value := StrToInt(Copy(TempString,25,Pos(',',TempString)-25));             {Addition of phage A at 3 times}
          Delete(TempString,1,Pos(',',TempString)+1);
          SpinEdit8.Value := StrToInt(Copy(TempString,1,Pos(',',TempString)-1));
          Delete(TempString,1,Pos(',',TempString)+1);
          SpinEdit9.Value := StrToInt(Copy(TempString,1));
          TempString := DataStrings.Strings[29];
          SpinEdit13.Value := StrToInt(Copy(TempString,25,Pos(',',TempString)-25));            {Addition of phage B at 3 times}
          Delete(TempString,1,Pos(',',TempString)+1);
          SpinEdit14.Value := StrToInt(Copy(TempString,1,Pos(',',TempString)-1));
          Delete(TempString,1,Pos(',',TempString)+1);
          SpinEdit15.Value := StrToInt(Copy(TempString,1));
          if Pos('Standard',DataStrings.Strings[32]) <> 0 then rgInfectionMod.ItemIndex := 0;  {Checking model modes}
          if Pos('Poisson',DataStrings.Strings[32]) <> 0 then rgInfectionMod.ItemIndex := 1;
          if Pos('Uninfected',DataStrings.Strings[33]) <> 0 then rgAdsorptionMod.ItemIndex := 0;
          if Pos('Susceptible',DataStrings.Strings[33]) <> 0 then rgAdsorptionMod.ItemIndex := 1;
          if Pos('Deterministic',DataStrings.Strings[34]) <> 0 then rgMutationMod.ItemIndex := 0;
          if Pos('Stochastic',DataStrings.Strings[34]) <> 0 then rgMutationMod.ItemIndex := 1;
          if Pos('Planktonic',DataStrings.Strings[35]) <> 0 then rgRefugeCells.ItemIndex := 0;
          if Pos('LIFO',DataStrings.Strings[35]) <> 0 then rgRefugeCells.ItemIndex := 1;
          FloatSpinEdit10.Value := StrToFloat(Copy(DataStrings.Strings[36],13,Pos(',',DataStrings.Strings[36])-13));            {Refuge cells rate in}
          FloatSpinEdit11.Value := StrToFloat(Copy(DataStrings.Strings[36],Pos(',',DataStrings.Strings[36])+8));                {Refuge cells rate out}
          if Pos('5 s',DataStrings.Strings[38]) <> 0 then rgTimeStep.ItemIndex := 0;                            {Set time step size}
          if Pos('15 s',DataStrings.Strings[38]) <> 0 then rgTimeStep.ItemIndex := 1;
          if Pos('30 s',DataStrings.Strings[38]) <> 0 then rgTimeStep.ItemIndex := 2;
          if Pos('1 min',DataStrings.Strings[38]) <> 0 then rgTimeStep.ItemIndex := 3;
          SpinEdit3.Value := StrToInt(Copy(DataStrings.Strings[39],35,Pos(',',DataStrings.Strings[39])-35));    {Set running time in hours}
          SpinEdit16.Value := StrToInt(Copy(DataStrings.Strings[39],Pos(',',DataStrings.Strings[39])+1));       {Set running time in minutes}

          if Pos(' 1 ',DataStrings.Strings[41]) <> 0 then SensitiveOut.Checked := true else SensitiveOut.Checked := false;
          if Pos(' 2 ',DataStrings.Strings[41]) <> 0 then ResistantAOut.Checked := true else ResistantAOut.Checked := false;
          if Pos(' 3 ',DataStrings.Strings[41]) <> 0 then ResistantBOut.Checked := true else ResistantBOut.Checked := false;
          if Pos(' 4 ',DataStrings.Strings[41]) <> 0 then ResistantABOut.Checked := true else ResistantABOut.Checked := false;
          if Pos(' 5 ',DataStrings.Strings[41]) <> 0 then InfectedAOut.Checked := true else InfectedAOut.Checked := false;
          if Pos(' 6 ',DataStrings.Strings[41]) <> 0 then InfectedBOut.Checked := true else InfectedBOut.Checked := false;
          if Pos(' 7 ',DataStrings.Strings[41]) <> 0 then InfectedABOut.Checked := true else InfectedABOut.Checked := false;
          if Pos(' 8 ',DataStrings.Strings[41]) <> 0 then AResInfbyBOut.Checked := true else AResInfbyBOut.Checked := false;
          if Pos(' 9 ',DataStrings.Strings[41]) <> 0 then BResInfbyAOut.Checked := true else BResInfbyAOut.Checked := false;
          if Pos(' 10 ',DataStrings.Strings[41]) <> 0 then RefugeOut.Checked := true else RefugeOut.Checked := false;
          if Pos(' 11 ',DataStrings.Strings[41]) <> 0 then PhageAOut.Checked := true else PhageAOut.Checked :=  false;
          if Pos(' 12 ',DataStrings.Strings[41]) <> 0 then PhageBOut.Checked := true else PhageBOut.Checked := false;
          if Pos(' 13 ',DataStrings.Strings[41]) <> 0 then ResourceCOut.Checked := true else ResourceCOut.Checked := false;
          if Pos(' 14 ',DataStrings.Strings[41]) <> 0 then LogScale.Checked := true else LogScale.Checked := false;
          if Pos(' 15 ',DataStrings.Strings[41]) <> 0 then RoundOff.Checked := true else RoundOff.Checked := false;

          StatusBar1.SimpleText := 'Data from file: ' + filename + '. Modify? / Press Run';
          end;
       end else ShowMessage('File error: This is not a Cocktail input file.');
  except
    on E:Exception do ShowMessage('File error: ' + E.Message);
  end; {try}
  FreeAndNil(DataStrings);

end;   {ReadCTLFile}


end.

