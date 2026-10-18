using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using NativeWebSocket;

[System.Serializable]
public class CommandData
{
    public string action;
    public string target;
    public float[] pos;
}

public class WebSocketClient : MonoBehaviour
{
    private WebSocket websocket;
    public GameObject targetCube; // Unity editöründen hareket ettirilecek Küp

    private bool isReconnecting = false;
    private bool isQuitting = false;

    async void Start()
    {
        await ConnectToWebSocket();
    }

    async Task ConnectToWebSocket()
    {
        if (isQuitting) return;

        websocket = new WebSocket("ws://localhost:8080");

        websocket.OnOpen += () =>
        {
            Debug.Log("[Unity] WebSocket Baðlantýsý Baþarýlý!");
            isReconnecting = false;
        };

        websocket.OnError += (e) =>
        {
            Debug.LogError("[Unity] WebSocket Hatasý: " + e);
        };

        websocket.OnClose += (e) =>
        {
            if (!isQuitting)
            {
                Debug.LogWarning("[Unity] WebSocket Baðlantýsý Koptu. Yeniden baðlanýlmaya çalýþýlýyor...");
                TryReconnect();
            }
        };

        websocket.OnMessage += (bytes) =>
        {
            var message = Encoding.UTF8.GetString(bytes);
            Debug.Log("[Unity] Gelen Mesaj: " + message);
            ProcessMessage(message);
        };

        await websocket.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif
    }

    private void ProcessMessage(string jsonMessage)
    {
        try
        {
            CommandData command = JsonUtility.FromJson<CommandData>(jsonMessage);

            if (command != null && command.action == "move")
            {
                if (targetCube != null && command.pos.Length == 3)
                {
                    // 1. Önce Gemini'den gelen delta/ofset deðerini alýyoruz
                    Vector3 offset = new Vector3(command.pos[0], command.pos[1], command.pos[2]);

                    // 2. Küpün MEVCUT konumuna bu ofseti ekliyoruz (Göreli Hareket)
                    targetCube.transform.position += offset;

                    Vector3 updatedPos = targetCube.transform.position;
                    Debug.Log($"[Unity] {command.target} objesi {offset} miktar kaydýrýldý. Yeni Güncel Konum: {updatedPos}");

                    // 3. Python'a yeni güncel konumu bildiriyoruz
                    SendMessageToPython($"{{\"status\": \"success\", \"moved\": \"{command.target}\", \"current_pos\": [{updatedPos.x}, {updatedPos.y}, {updatedPos.z}]}}");
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[Unity] Mesaj iþlenirken hata oluþtu: " + ex.Message);
        }
    }

    public async void SendMessageToPython(string message)
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
        {
            await websocket.SendText(message);
        }
    }

    private async void TryReconnect()
    {
        if (isReconnecting || isQuitting) return;
        isReconnecting = true;

        while (!isQuitting && (websocket == null || websocket.State != WebSocketState.Open))
        {
            Debug.Log("[Unity] 3 saniye sonra tekrar baðlanýlacak...");
            await Task.Delay(3000);
            if (isQuitting) break;
            await ConnectToWebSocket();
        }
    }

    private async Task CloseWebSocket()
    {
        isQuitting = true;
        if (websocket != null)
        {
            if (websocket.State == WebSocketState.Open)
            {
                await websocket.Close();
            }
            websocket = null;
        }
    }

    private async void OnDisable()
    {
        await CloseWebSocket();
    }

    private async void OnApplicationQuit()
    {
        await CloseWebSocket();
    }
}