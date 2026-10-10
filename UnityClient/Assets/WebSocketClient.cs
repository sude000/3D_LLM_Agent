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

    async void Start()
    {
        await ConnectToWebSocket();
    }

    async Task ConnectToWebSocket()
    {
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
            Debug.LogWarning("[Unity] WebSocket Baðlantýsý Koptu. Yeniden baðlanýlmaya çalýþýlýyor...");
            TryReconnect();
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
                    Vector3 newPos = new Vector3(command.pos[0], command.pos[1], command.pos[2]);
                    targetCube.transform.position = newPos;
                    Debug.Log($"[Unity] {command.target} objesi {newPos} konumuna taþýndý.");

                    SendMessageToPython($"{{\"status\": \"success\", \"moved\": \"{command.target}\"}}");
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
        if (websocket.State == WebSocketState.Open)
        {
            await websocket.SendText(message);
        }
    }

    private async void TryReconnect()
    {
        if (isReconnecting) return;
        isReconnecting = true;

        while (websocket == null || websocket.State != WebSocketState.Open)
        {
            Debug.Log("[Unity] 3 saniye sonra tekrar baðlanýlacak...");
            await Task.Delay(3000);
            await ConnectToWebSocket();
        }
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }
}