import sys

from std_msgs.msg import String
import rclpy

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


msg = """
This node takes keypresses from the keyboard and publishes the desired
mode for the magmap node.
---------------------------
1: train
2: test
3: none
---------------------------
CTRL-C to quit
"""

modeBindings = {
    '1': 'train',
    '2': 'test',
    '3': 'none',
}


def getKey(settings):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main():
    settings = saveTerminalSettings()

    rclpy.init()

    node = rclpy.create_node('magmap_control')
    pub = node.create_publisher(String, 'magmap/mode', 10)

    try:
        print(msg)
        
        while True:
            key = getKey(settings)
            if key in modeBindings.keys():
                mode_msg = String()
                mode_msg.data = modeBindings[key]
                pub.publish(mode_msg)

            if (key == '\x03'):
                break

    except Exception as e:
        print(e)

    finally:

        restoreTerminalSettings(settings)


if __name__ == '__main__':
    main()
