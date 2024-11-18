#!/usr/bin/env python3

from fastapi import FastAPI, Request, Query, HTTPException
from pydantic import BaseModel, TypeAdapter, Field, field_validator
from typing import List, Union, Literal, Annotated, Optional
from datetime import datetime, timezone
import pandas as pd


class TimeseriesQuery_Params(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}
    from_: Optional[float] = Field(None, ge=0, alias='from')
    to: Optional[float] = Field(None, ge=0)
    interval: Optional[int] = Field(None, ge=0)
    limit: Optional[int] = Field(1_000, gt=0, le=10_000)
    fields: Optional[str] = Field('*')
    
#    @field_validator("fields")
#    def validate_option(cls, v):
#        available_fields = ['*', 'efc', 'offset', 'status', 'tracking_cnt', 'phase', 'gps_time_valid']
#        assert all([x.strip() in available_fields for x in v.lower().split(',')])


class GPSDO_Response(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}
    utc: float
    offset: Optional[float] = Field(None)
    efc: Optional[float] = Field(None, ge=0)
    status: Optional[int] = Field(None, ge=0)
    phase: Optional[float] = Field(None)
    gps_time_valid: Optional[int] = Field(None)
    tracking_cnt: Optional[int] = Field(None, gt=0, le=10_000)
    gps_lock: Optional[int] = Field(None)
    out_ref: Optional[str] = Field(None)

gpsdo_response_adapter = TypeAdapter(List[GPSDO_Response])

app = FastAPI()

@app.get("/")
async def root():
    return {"api": "tdc API v1.2.1"}


@app.get("/api/tdc/query" , response_model=List[GPSDO_Response], response_model_exclude_none=True)
async def query(q: Annotated[TimeseriesQuery_Params, Query()]):
    try:
        if q:
            print('q', q)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Invalid input parameters ({type(e)}): {e}")
    try:
        uc_cols = ['efc', 'status', 'tracking_cnt', 'phase', 'gps_time_valid', 'gps_lock', 'out_ref']
        td = pd.DataFrame(columns=['utc', 'offset'])
        uc = pd.DataFrame(columns=['utc', 'gps_time', 'gps_date'] + uc_cols)
        if any([f in q.fields for f in ['*', 'offset']]):
            f = '/home/md/time/tts4/tdc_ti7200/data/Trimble_vs_M8T-60631.txt'
            td = pd.read_csv(f, sep='\t')
            td = td[['UTC', 'TI']]
        if any([f in q.fields for f in ['*']+uc_cols]):            
            f = '/home/md/time/uccm/uccm.txt'
            uc = pd.read_json(f, lines=True)
            uc = uc[['gps_date', 'gps_time']+uc_cols]    # limit the data only to known columns (not requested)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Input data not accessible ({type(e)}): {e}")

    try:
        if len(uc) > 0:
            uc['status'] = uc.status.str[9:-1]
            uc['utc'] = pd.to_datetime(uc.gps_date + ' ' + uc.gps_time.dt.time.astype(str), utc=True)
            uc['utc'] = uc.utc.astype(int) // 10**9
            uc['gps_time_valid'] = (uc.gps_time_valid == '').astype('int')        

        if len(td) > 0:
            td = td.loc[td.TI.notna()]
            td['UTC'] = td.UTC.astype(int)
            td.rename(columns={'UTC': 'utc', 'TI': 'offset'}, inplace=True)

        if q.from_:
            td = td.loc[td.utc >= q.from_/1000]
            uc = uc.loc[uc.utc >= q.from_/1000]
        if q.to:
            td = td.loc[td.utc <= q.to/1000]
            uc = uc.loc[uc.utc <= q.to/1000]
        df = pd.merge(uc, td, on='utc', how='outer')
        if ('*' not in q.fields):
            flds = [x.strip() for x in q.fields.lower().split(',')]
            df = df[['utc']+flds]
        if q.limit:
            df = df.tail(q.limit)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Internal processing error ({type(e)}): {e}")

    try:
        ret = gpsdo_response_adapter.validate_json(df.to_json(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Internal data type error ({type(e)}): {e}")

    return ret
