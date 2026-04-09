package main

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	pb "github.com/hyperledger/fabric-protos-go/peer"
)

type TrustRegistryChaincode struct{}

type ModelRecord struct {
	Round          int     `json:"round"`
	ModelHash      string  `json:"modelHash"`
	BlockHash      string  `json:"blockHash"`
	PrevBlockHash  string  `json:"prevBlockHash"`
	GlobalF1       float64 `json:"globalF1"`
	GlobalAUC      float64 `json:"globalAUC"`
	TrustedClients []int   `json:"trustedClients"`
	FlaggedClients []int   `json:"flaggedClients"`
	ParamCount     int     `json:"paramCount"`
	TotalBytes     int     `json:"totalBytes"`
	Timestamp      int64   `json:"timestamp"`
	TxID           string  `json:"txId"`
}

type TamperAlert struct {
	ID        int    `json:"id"`
	Round     int    `json:"round"`
	AlertType string `json:"alertType"`
	Detail    string `json:"detail"`
	Severity  string `json:"severity"`
	Timestamp int64  `json:"timestamp"`
	TxID      string `json:"txId"`
}

type AuditEvent struct {
	ID        int    `json:"id"`
	EventType string `json:"eventType"`
	Round     int    `json:"round"`
	Actor     string `json:"actor"`
	Data      string `json:"data"`
	Timestamp int64  `json:"timestamp"`
	TxID      string `json:"txId"`
}

type VerifyResult struct {
	Verified    bool   `json:"verified"`
	Round       int    `json:"round"`
	StoredHash  string `json:"storedHash"`
	ClaimedHash string `json:"claimedHash"`
}

const genesisZeroHash = "0000000000000000000000000000000000000000000000000000000000000000"

func success(payload []byte) pb.Response {
	if payload == nil {
		payload = []byte("{}")
	}
	return pb.Response{Status: 200, Payload: payload}
}

func failure(err error) pb.Response {
	return pb.Response{Status: 500, Message: err.Error()}
}

func txMillis(stub shim.ChaincodeStubInterface) int64 {
	ts, err := stub.GetTxTimestamp()
	if err != nil || ts == nil {
		return 0
	}
	return ts.Seconds*1000 + int64(ts.Nanos)/1_000_000
}

func (t *TrustRegistryChaincode) Init(stub shim.ChaincodeStubInterface) pb.Response {
	return success(nil)
}

func (t *TrustRegistryChaincode) Invoke(stub shim.ChaincodeStubInterface) pb.Response {
	fn, args := stub.GetFunctionAndParameters()

	switch fn {
	case "RegisterModel":
		if len(args) != 10 {
			return failure(fmt.Errorf("RegisterModel expects 10 args"))
		}
		return t.registerModel(stub, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9])
	case "GetModel":
		if len(args) != 1 {
			return failure(fmt.Errorf("GetModel expects 1 arg"))
		}
		return t.getModel(stub, args[0])
	case "VerifyModelHash":
		if len(args) != 2 {
			return failure(fmt.Errorf("VerifyModelHash expects 2 args"))
		}
		return t.verifyModelHash(stub, args[0], args[1])
	case "QueryAllModels":
		return t.queryAllModels(stub)
	case "RaiseTamperAlert":
		if len(args) != 4 {
			return failure(fmt.Errorf("RaiseTamperAlert expects 4 args"))
		}
		return t.raiseTamperAlert(stub, args[0], args[1], args[2], args[3])
	case "GetAlerts":
		return t.getAlerts(stub)
	case "AppendEvent":
		if len(args) != 4 {
			return failure(fmt.Errorf("AppendEvent expects 4 args"))
		}
		return t.appendEvent(stub, args[0], args[1], args[2], args[3])
	case "ExportAuditTrail":
		return t.exportAuditTrail(stub)
	default:
		return failure(fmt.Errorf("unknown function: %s", fn))
	}
}

func (t *TrustRegistryChaincode) nextID(stub shim.ChaincodeStubInterface, ns string) (int, error) {
	key := "CTR:" + ns
	b, err := stub.GetState(key)
	if err != nil {
		return 0, err
	}
	n := 0
	if b != nil {
		n, _ = strconv.Atoi(string(b))
	}
	n++
	return n, stub.PutState(key, []byte(strconv.Itoa(n)))
}

func (t *TrustRegistryChaincode) registerModel(stub shim.ChaincodeStubInterface,
	roundStr, modelHash, blockHash, prevBlockHash, f1Str, aucStr,
	trustedJSON, flaggedJSON, paramStr, bytesStr string) pb.Response {

	round, _ := strconv.Atoi(roundStr)
	key := "MODEL:" + roundStr
	if ex, _ := stub.GetState(key); ex != nil {
		return failure(fmt.Errorf("round %d already registered", round))
	}
	if round == 1 {
		if prevBlockHash != "GENESIS" && prevBlockHash != genesisZeroHash {
			return failure(fmt.Errorf("round 1 must use GENESIS"))
		}
	} else {
		prevBytes, _ := stub.GetState(fmt.Sprintf("MODEL:%d", round-1))
		if prevBytes == nil {
			return failure(fmt.Errorf("round %d not found - register sequentially", round-1))
		}
		var prev ModelRecord
		json.Unmarshal(prevBytes, &prev)
		if prev.BlockHash != prevBlockHash {
			return failure(fmt.Errorf("CHAIN_VIOLATION round %d", round))
		}
	}
	f1, _ := strconv.ParseFloat(f1Str, 64)
	auc, _ := strconv.ParseFloat(aucStr, 64)
	pc, _ := strconv.Atoi(paramStr)
	tb, _ := strconv.Atoi(bytesStr)
	var trusted, flagged []int
	json.Unmarshal([]byte(trustedJSON), &trusted)
	json.Unmarshal([]byte(flaggedJSON), &flagged)
	rec := ModelRecord{Round: round, ModelHash: modelHash, BlockHash: blockHash,
		PrevBlockHash: prevBlockHash, GlobalF1: f1, GlobalAUC: auc,
		TrustedClients: trusted, FlaggedClients: flagged, ParamCount: pc, TotalBytes: tb,
		Timestamp: txMillis(stub), TxID: stub.GetTxID()}
	b, _ := json.Marshal(rec)
	if err := stub.PutState(key, b); err != nil {
		return failure(err)
	}
	return success([]byte(`{"ok":true}`))
}

func (t *TrustRegistryChaincode) getModel(stub shim.ChaincodeStubInterface, roundStr string) pb.Response {
	b, err := stub.GetState("MODEL:" + roundStr)
	if err != nil || b == nil {
		out, _ := json.Marshal(map[string]interface{}{
			"found": false,
			"round": roundStr,
		})
		return success(out)
	}
	return success(b)
}

func (t *TrustRegistryChaincode) verifyModelHash(stub shim.ChaincodeStubInterface, roundStr, claimed string) pb.Response {
	b, err := stub.GetState("MODEL:" + roundStr)
	if err != nil || b == nil {
		out, _ := json.Marshal(&VerifyResult{Verified: false})
		return success(out)
	}
	var rec ModelRecord
	json.Unmarshal(b, &rec)
	round, _ := strconv.Atoi(roundStr)
	out, _ := json.Marshal(&VerifyResult{Verified: rec.ModelHash == claimed, Round: round, StoredHash: rec.ModelHash, ClaimedHash: claimed})
	return success(out)
}

func (t *TrustRegistryChaincode) queryAllModels(stub shim.ChaincodeStubInterface) pb.Response {
	iter, err := stub.GetStateByRange("MODEL:", "MODEL:~")
	if err != nil {
		return failure(err)
	}
	defer iter.Close()
	var out []*ModelRecord
	for iter.HasNext() {
		kv, _ := iter.Next()
		var rec ModelRecord
		if json.Unmarshal(kv.Value, &rec) == nil {
			out = append(out, &rec)
		}
	}
	b, _ := json.Marshal(out)
	return success(b)
}

func (t *TrustRegistryChaincode) raiseTamperAlert(stub shim.ChaincodeStubInterface, roundStr, alertType, detail, severity string) pb.Response {
	id, err := t.nextID(stub, "ALERT")
	if err != nil {
		return failure(err)
	}
	round, _ := strconv.Atoi(roundStr)
	alert := TamperAlert{ID: id, Round: round, AlertType: alertType, Detail: detail, Severity: severity, Timestamp: txMillis(stub), TxID: stub.GetTxID()}
	b, _ := json.Marshal(alert)
	if err := stub.PutState(fmt.Sprintf("ALERT:%08d", id), b); err != nil {
		return failure(err)
	}
	return success([]byte(`{"ok":true}`))
}

func (t *TrustRegistryChaincode) getAlerts(stub shim.ChaincodeStubInterface) pb.Response {
	iter, err := stub.GetStateByRange("ALERT:", "ALERT:~")
	if err != nil {
		return failure(err)
	}
	defer iter.Close()
	var out []*TamperAlert
	for iter.HasNext() {
		kv, _ := iter.Next()
		var a TamperAlert
		if json.Unmarshal(kv.Value, &a) == nil {
			out = append(out, &a)
		}
	}
	b, _ := json.Marshal(out)
	return success(b)
}

func (t *TrustRegistryChaincode) appendEvent(stub shim.ChaincodeStubInterface, eventType, roundStr, actor, dataJSON string) pb.Response {
	id, err := t.nextID(stub, "AUDIT")
	if err != nil {
		return failure(err)
	}
	round, _ := strconv.Atoi(roundStr)
	ev := AuditEvent{ID: id, EventType: eventType, Round: round, Actor: actor, Data: dataJSON, Timestamp: txMillis(stub), TxID: stub.GetTxID()}
	b, _ := json.Marshal(ev)
	if err := stub.PutState(fmt.Sprintf("AUDIT:%08d", id), b); err != nil {
		return failure(err)
	}
	return success([]byte(`{"ok":true}`))
}

func (t *TrustRegistryChaincode) exportAuditTrail(stub shim.ChaincodeStubInterface) pb.Response {
	iter, err := stub.GetStateByRange("AUDIT:", "AUDIT:~")
	if err != nil {
		return failure(err)
	}
	defer iter.Close()
	var out []*AuditEvent
	for iter.HasNext() {
		kv, _ := iter.Next()
		var ev AuditEvent
		if json.Unmarshal(kv.Value, &ev) == nil {
			out = append(out, &ev)
		}
	}
	b, _ := json.Marshal(out)
	return success(b)
}

func main() {
	if err := shim.Start(new(TrustRegistryChaincode)); err != nil {
		panic(err)
	}
}
