unit Globalvariables;

{$mode objfpc}{$H+}

interface

  uses
    SysUtils;
    procedure ClearVariables;
    procedure ClearOldSet;

  var
     { Bacteria }
    GrowingCells, Sensitive, ResistantToA, ResistantToB, ResistantToAB, DecayedBact,
    InfectedA, InfectedB, InfectedAB, InfectedAB1, InfectedAB2,
    InfectedAB3, AResInfectedB, BResInfectedA, UnInfectedA, UnInfectedB, AllInfectedA, AllInfectedB,
    NewInfectedA, NewInfectedB, NewInfectedAB, NewInfectedAB1, NewInfectedAB2, NewInfectedAB3,
    NewAResInfectedB, NewBResInfectedA,
    FreqNewInfectedA, FreqNewInfectedB, FreqNewInfectedAB, FreqNewSensInfectedA, FreqNewInfectedAB1,
    FreqNewBResInfectedA, FreqNewSensInfectedB, FreqNewSensInfectedAB, FreqNewInfectedAB2, FreqNewAResInfectedB,
    NewInfectedAonBA, NewInfectedBonAB, TotalBacteria, AllSusceptible,
    GrowingBacteria, NewSensitive,
    NewResistantToA, NewResistantToB, NewResistantToAB, NewMutations, SumOfNewSensitiveMutations,
    AccResistantToA, AccResistantToB, AccResistantToAB, AccSensitive, RefugeRateIn, RefugeRateOut,
    TotalRefugeRateOut, TotalRefugeCells,
    { Parameters }
    MaxGrowthSens, MaxGrowthResA, MaxGrowthResB, MaxGrowthResAB, MonodK, MonodFunc,
    ConversEff, DecayRateBact, FlowRate, ResistRateA, ResistRateB, ResistRateAB,
    StartTitre, ResevoirC, ResourceC, ResourceStartC,

    { Phages }
    AdsorbRateA, AdsorbRateB, DecayRatePhageA, DecayRatePhageB, PhageTitreA, PhageTitreB,
    BoundPhagesA, BoundPhagesB, FreePhagesA, FreePhagesB, TimeOut: Real;

    { Parameters }
    LatencyA, LatencyB, LatencyAB, BurstA, BurstB, RunningTimeHour, RunningTimeMinute, RunningTime,
    AddTimeOneA, AddTimeTwoA, AddTimeThreeA, AddTimeOneB, AddTimeTwoB, AddTimeThreeB,
    HourDivisor, MinuteDivisor: Integer;

    ConveyorA, ConveyorB, ConveyorAB, ConveyorBResA, ConveyorAResB, ConveyorAonBA, ConveyorAonBB,
    ConveyorBonAA, ConveyorBonAB, ConveyorBoundA, ConveyorBoundB, ConveyorFreqInfectedB,
    AckRefugeSensitive, AckRefugeResistantToA, AckRefugeResistantToB, AckRefugeResistantToAB, AckRefugeCells: array of Real;

    ITimeToLysisA, ITimeToLysisB: array of integer;

    ICount, ITimeA, ITimeB, ITimeAB, ITimeBResA, ITimeAResB, ITimeABA, ITimeBAB, ITimeBAA, ITimeABB,
    ITimeBoundA, ITimeBoundB, IRefugePopNr, ICountTimeSteps: Integer;

    PhageInfection, ReleaseA, ReleaseB, ReleaseAB, ReleaseBResA, ReleaseAResB, ReleaseABA, ReleaseBAB,
    ReleaseBAA, ReleaseABB, CountToTen, DoLog: Boolean;

    OutputParam: String;

implementation

    procedure ClearVariables;
      begin
      { Bacteria }
        GrowingCells := 0; Sensitive := 0; ResistantToA := 0; ResistantToB := 0; ResistantToAB := 0; AllSusceptible := 0; DecayedBact := 0;
        InfectedA := 0; InfectedB := 0; InfectedAB := 0; InfectedAB1 := 0; InfectedAB2 := 0;
        InfectedAB3 := 0; AResInfectedB := 0; BResInfectedA := 0; UnInfectedA := 0; UnInfectedB := 0; AllInfectedA := 0; AllInfectedB := 0;
        NewInfectedA := 0; NewInfectedB := 0; NewInfectedAB := 0; NewInfectedAB1 := 0; NewInfectedAB2 := 0; NewInfectedAB3 := 0;
        NewAResInfectedB := 0; NewBResInfectedA := 0;
        FreqNewInfectedA := 0; FreqNewInfectedB := 0; FreqNewInfectedAB := 0; FreqNewSensInfectedA := 0; FreqNewInfectedAB1 := 0;
        FreqNewBResInfectedA := 0; FreqNewSensInfectedB := 0; FreqNewSensInfectedAB := 0; FreqNewInfectedAB2 := 0; FreqNewAResInfectedB := 0;
        NewInfectedAonBA := 0; NewInfectedBonAB := 0; TotalBacteria := 0; GrowingBacteria := 0; NewSensitive := 0;
        NewResistantToA := 0; NewResistantToB := 0; NewResistantToAB := 0; NewMutations := 0; SumOfNewSensitiveMutations := 0;
        AccResistantToA := 0; AccResistantToB := 0; AccResistantToAB := 0; AccSensitive := 0;
        RefugeRateIn := 0; RefugeRateOut := 0; TotalRefugeCells :=0;
        { Parameters }
        MaxGrowthSens := 0; MaxGrowthResA := 0; MaxGrowthResB := 0; MaxGrowthResAB := 0; MonodK := 0; MonodFunc := 0;
        ConversEff := 0; DecayRateBact := 0; FlowRate := 0; ResistRateA := 0; ResistRateB := 0; ResistRateAB := 0;
        StartTitre := 0; ResevoirC := 0; ResourceC := 0; ResourceStartC := 0;
        { Phages }
        AdsorbRateA := 0; AdsorbRateB := 0; DecayRatePhageA := 0; DecayRatePhageB := 0; PhageTitreA := 0; PhageTitreB := 0;
        BoundPhagesA := 0; BoundPhagesB := 0; FreePhagesA := 0; FreePhagesB := 0; TimeOut := 0;

        LatencyA := 0; LatencyB := 0; LatencyAB := 0; BurstA := 0; BurstB := 0; RunningTimeHour := 0; RunningTimeMinute := 0; RunningTime := 0;
        AddTimeOneA := 0; AddTimeTwoA := 0; AddTimeThreeA := 0; AddTimeOneB := 0; AddTimeTwoB := 0; AddTimeThreeB := 0;
        HourDivisor := 0; MinuteDivisor := 0;

        ICount := 0; ITimeA := 0; ITimeB := 0; ITimeAB := 0; ITimeBResA := 0; ITimeAResB := 0; ITimeABA := 0; ITimeBAB := 0; ITimeBAA := 0;
        ITimeABB := 0; ITimeBoundA := 0; ITimeBoundB := 0; IRefugePopNr := 0; ICountTimeSteps := 0;
      end;

      procedure ClearOldSet;
      begin
        {Bacteria}
        ResistantToA := Sensitive * ResistantToA;
        ResistantToB  := Sensitive * ResistantToB;
        ResistantToAB  := Sensitive * ResistantToAB;
        ResistRateAB := ResistRateA * ResistRateB;
        UnInfectedA := Sensitive;
        UnInfectedB := Sensitive;
        AllInfectedA := 0;
        AllInfectedB := 0;
        Sensitive := Sensitive - ResistantToA - ResistantToB - ResistantToAB;
        AccResistantToA := 0;
        AccResistantToB := 0;
        AccResistantToAB := 0;
        InfectedA := 0;
        InfectedB := 0;
        InfectedAB := 0;
        InfectedAB1 :=0;
        InfectedAB2 :=0;
        InfectedAB3 :=0;
        AResInfectedB := 0;
        BResInfectedA := 0;
        NewInfectedA := 0;
        NewInfectedB := 0;
        NewInfectedAB := 0;
        NewSensitive := 0;
        NewAResInfectedB := 0;
        NewBResInfectedA := 0;
        NewInfectedAonBA := 0;
        NewInfectedBonAB := 0;
        FreqNewSensInfectedA := 0;
        FreqNewInfectedAB1 := 0;
        FreqNewBResInfectedA := 0;
        FreqNewSensInfectedB := 0;
        FreqNewInfectedAB2 := 0;
        FreqNewAResInfectedB := 0;

        ITimeA := 0;
        ITimeB := 0;
        ITimeAB := 0;
        ITimeBResA := 0;
        ITimeAResB := 0;
        ITimeABA := 0;
        ITimeABB := 0;
        ITimeBAB := 0;
        ITimeBAA := 0;
        ITimeBoundA := 0;
        ITimeBoundB := 0;

        {Phages}
        BoundPhagesA := 0;
        BoundPhagesB := 0;

        FreePhagesA := 0;
        FreePhagesB := 0;

        if (PhageTitreA >= 1) or (PhageTitreB >=1) then PhageInfection := true else PhageInfection := false;
        ReleaseA := false;
        ReleaseB := false;
        ReleaseAB := false;
        ReleaseABA := false;
        ReleaseBAB := false;
        ReleaseABB := false;
        ReleaseBAA := false;
        ReleaseBResA := false;
        ReleaseAResB := false;
        CountToTen := false;
        DoLog := false;
        OutputParam := ('');
      end;

end.

